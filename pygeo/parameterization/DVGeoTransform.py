# Standard Python modules
from collections import OrderedDict
import copy

# External modules
from baseclasses.utils import Error
from mpi4py import MPI
import numpy as np
from scipy import sparse

from .designVars import geoDVTransformation


# TODO inherit from basedvgeometry?
class DVGeometryTransform:
    """
    A class for manipulating multiple components using multiple FFDs
    and handling design changes near component intersections.

    Parameters
    ----------
    DVGeoTop : DVGeometry
        The top level DVGeometry object that implements geometric design changes after the transformations

    name : str
        This is prepended to every DV name for ensuring design variables names are
        unique to pyOptSparse.
        This can be disabled

    checkDVs : bool, optional
        Flag to check whether there are duplicate DV names in or across components.

    isComplex : bool, optional
        Flag to use complex variables for complex step verification.

    """

    def __init__(self, DVGeoTop, name=None, checkDVs=True, isComplex=False):

        self.name = name
        self.DVGeoTop = DVGeoTop
        self.transormationFuncs = OrderedDict()
        self.points = OrderedDict()
        self.ptSetNames = []
        self.updated = {}
        self.checkDVs = checkDVs
        self.isComplex = isComplex

    def addTransformationFunction(self, funcName, funcCallback):

        # save the info in the dictionary
        self.transormationFuncs[funcName] = TransformationFunc(funcName, funcCallback)

    def addTransformationDV(self, funcName, dvName, value, lower=None, upper=None, scale=1.0, config=None, prependName=True):
        self.transormationFuncs[funcName].addDV(dvName, value, lower, upper, scale, config, prependName)

    def addPointSet(self, points, ptName, transformationFunc=None, config=None, **kwargs):

        # make the input at least 2d in case a single test point is passed
        points = np.atleast_2d(points)

        # we run the transformation function before passing to DVGeo Top
        if transformationFunc is not None:
            pointsBase = self.transormationFuncs[transformationFunc].update(points, "bwd", config=config)
        else:
            pointsBase = points

        # save the ptset information
        self.points[ptName] = PointSet(pointsBase, transformationFunc)

        # next, we forward the ptset to the top level DVGeo
        self.DVGeoTop.addPointSet(pointsBase, ptName, **kwargs)

        self.updated[ptName] = False

    def setDesignVars(self, dvDict):
        """
        Standard routine for setting design variables from a design variable dictionary.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables.
            The keys of the dictionary must correspond to the design variable names.
            Any additional keys in the dictionary are simply ignored.

        """

        # Check if we have duplicate DV names
        if self.checkDVs:
            dvNames = self.getVarNames()
            duplicates = len(dvNames) != len(set(dvNames))
            if duplicates:
                raise Error(
                    "There are duplicate DV names in a component or across components. "
                    "If this is intended, initialize the DVGeometryMulti class with checkDVs=False."
                )

        # loop over the transformation funcs and set DVs
        for transFunc in self.transormationFuncs.values():
            transFunc.setDesignVars(dvDict)

        # also set the DVs in the top level DVGeo
        self.DVGeoTop.setDesignVars(dvDict)

        # Flag all the pointSets as not being up to date:
        for pointSet in self.updated:
            self.updated[pointSet] = False

    def getValues(self):
        """
        Generic routine to return the current set of design variables.
        Values are returned in a dictionary format that would be suitable for a subsequent call to setDesignVars().

        Returns
        -------
        dvDict : dict
            Dictionary of design variables.

        """

        # start with the dvs of the top dvgeo
        dvDict = self.DVGeoTop.getValues()

        # loop over the transformation funcs, and update the dict
        for transFunc in self.transormationFuncs.values():
            dvDict.update(transFunc.getValues())

        return dvDict

    def update(self, ptSetName, config=None):
        """
        This is the main routine for returning coordinates that have been updated by design variables.
        Multiple configs are not supported.

        Parameters
        ----------
        ptSetName : str
            Name of point set to return.
            This must match one of those added in an :func:`addPointSet()` call.

        """

        points = self.DVGeoTop.update(ptSetName, config=config)

        # finally, apply the transformation if this ptset has any
        transFunc = self.points[ptSetName].transformationFunc
        if transFunc is not None:
            points = self.transormationFuncs[transFunc].update(points, "fwd", config=config)

        # Finally flag this pointSet as being up to date:
        self.updated[ptSetName] = True

        return points

    def pointSetUpToDate(self, ptSetName):
        """
        This is used externally to query if the object needs to update its point set or not.
        When update() is called with a point set, the self.updated value for pointSet is flagged as True.
        We reset all flags to False when design variables are set because nothing (in general) will up to date anymore.
        Here we just return that flag.

        Parameters
        ----------
        ptSetName : str
            The name of the pointset to check.

        """
        if ptSetName in self.updated:
            return self.updated[ptSetName]
        else:
            return True

    def getNDV(self):
        """Return the number of DVs."""
        # Loop over components and sum the number of DVs
        nDV = self.DVGeoTop.getNDV()
        for transFunc in self.transormationFuncs.values():
            nDV += transFunc.getNDV()

        return nDV

    def getVarNames(self, pyOptSparse=False):
        """
        Return a list of the design variable names.
        This is typically used when specifying a ``wrt=`` argument for pyOptSparse.

        Examples
        --------
        >>> optProb.addCon(.....wrt=DVGeo.getVarNames())

        """
        dvNames = self.DVGeoTop.getVarNames()
        for transFunc in self.transormationFuncs.values():
            dvNames.extend(transFunc.getVarNames())

        return dvNames

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        """


        """
        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])
        N = dIdpt.shape[0]

        dIdxDict = {}

        # first, process our own derivatives based on dIdpt. We do this first because the top level dvgeo might modify this
        transFuncName = self.points[ptSetName].transformationFunc
        if transFuncName is not None:
            transFunc = self.transormationFuncs[transFuncName]
            pointsBase = self.points[ptSetName].pointsBase
            dPtdDV = transFunc.getJacobian(pointsBase, config=config)
            # save the jacobian. this will be called more times
            self.points[ptSetName].dPtdDV = dPtdDV

            dIdx_local = dPtdDV.T.dot(dIdpt)

            # multiply with didpt
            if comm:  # If we have a comm, globaly reduce with sum
                dIdxArray = comm.allreduce(dIdx_local, op=MPI.SUM)
            else:
                dIdxArray = dIdx_local

            # Now convert to dict and save
            dIdxDict.update(self.convertSensitivityToDict(dIdxArray))

            # rotate the didpt so that its in dvgeo reference
            for ifunc in range(N):
                dIdpt[ifunc] = transFunc.sens(dIdpt[ifunc])


        # process the top dvgeo derivatives
        # combine dictionaries and return
        dIdxDict.update(self.DVGeoTop.totalSensitivity(dIdpt, ptSetName, comm=comm, config=config))

        return dIdxDict

    def addVariablesPyOpt(
        self,
        optProb,
        globalVars=True,
        localVars=True,
        sectionlocalVars=True,
        ignoreVars=None,
        freezeVars=None,
        comps=None,
    ):
        """
        Add the current set of variables to the optProb object.

        Parameters
        ----------
        optProb : pyOpt_optimization class
            Optimization problem definition to which variables are added

        globalVars : bool
            Flag specifying whether global variables are to be added

        localVars : bool
            Flag specifying whether local variables are to be added

        ignoreVars : list of strings
            List of design variables the user doesn't want to use
            as optimization variables.

        freezeVars : list of string
            List of design variables the user wants to add as optimization
            variables, but to have the lower and upper bounds set at the current
            variable. This effectively eliminates the variable, but it the variable
            is still part of the optimization.

        comps : list
            List of components we want to add the DVs of.
            If no list is provided, we will add DVs from all components.

        """

        # add the top level dvgeo first
        self.DVGeoTop.addVariablesPyOpt(
                optProb,
                globalVars=globalVars,
                localVars=localVars,
                sectionlocalVars=sectionlocalVars,
                ignoreVars=ignoreVars,
                freezeVars=freezeVars,
            )

        # then add the transformation function DVs
        for transformFunc in self.transormationFuncs.values():
            transformFunc.addVariablesPyOpt(optProb, freezeVars)


    def getLocalIndex(self, iVol):
        """Return the local index mapping that points to the global coefficient list for a given volume.

        Parameters
        ----------

        iVol : int
            Index specifying the FFD volume.

        comp : str
            Name of the component.

        """

        # Call this on the component DVGeo
        return self.DVGeoTop.FFD.topo.lIndex[iVol].copy()


class TransformationFunc:
    def __init__(self, name, funcCallback):

        self.dvDict = OrderedDict()
        self.dvList = []
        self.funcCallback = funcCallback

        self.name = name

    def addDV(self, dvName, value, lower, upper, scale, config, prependName):
        # if the parent DVGeometry object has a name attribute, prepend it
        if self.name is not None and prependName:
            dvName = self.name + "_" + dvName

        self.dvDict[dvName] = geoDVTransformation(dvName, value, lower, upper, scale, config)
        self.dvList.append(dvName)

    def setDesignVars(self, dvDict):
        def _checkArrLength(key, nIn, nRef):
            if nIn != nRef:
                raise Error(
                    f"Incorrect number of design variables for DV: {key}.\n"
                    + f"Expecting {nRef} variables but received {nIn}"
                )

        for key in dvDict:
            if key in self.dvDict:
                vals_to_set = np.atleast_1d(dvDict[key]).astype("D")
                _checkArrLength(key, len(vals_to_set), self.dvDict[key].nVal)
                self.dvDict[key].value = vals_to_set

    def getValues(self):
        """
        Generic routine to return the current set of design
        variables. Values are returned in a dictionary format
        that would be suitable for a subsequent call to :func:`setDesignVars`

        Returns
        -------
        dvDict : dict
            Dictionary of design variables
        """

        dvDict = {}
        for key in self.dvList:
            dvDict[key] = self.dvDict[key].value
        return dvDict

    def update(self, points, mode, config=None):
        # call the function callback with the current DVs and config
        return self.funcCallback(points, mode=mode, applyDisplacement=True, config=config, dvDict=self.getValues())

    def getNDV(self):
        nDV = 0
        for dvName in self.dvList:
            nDV += self.dvDict[dvName].nVal
        return nDV

    def getVarNames(self):
        return self.dvList

    def getJacobian(self, pointsBase, config=None):
        # get the dv jacobian for this ptset
        dh = 1e-40
        xdv = copy.deepcopy(self.getValues())
        # allocate the jacobian
        npts = pointsBase.shape[0]
        dPtdDV = np.zeros((self.getNDV(), npts, 3))

        # loop over the DVs and perturb each
        ii = 0
        for key in self.dvList:
            dv = self.dvDict[key]
            for jj in range(dv.nVal):
                # copy the current value
                refVal = xdv[key][jj]
                # perturb
                xdv[key][jj] = refVal + dh * 1j
                # evaluate
                pointsPlus = self.funcCallback(pointsBase, mode="fwd", applyDisplacement=True, config=config, dvDict=xdv)
                # set in the jacobian matrix
                dPtdDV[ii + jj] = np.imag(pointsPlus) / dh
                # reset the value
                xdv[key][jj] = refVal

            ii += dv.nVal

        return dPtdDV

    def convertSensitivityToDict(self, dIdx, out1D=False):
        """
        This function takes the result of totalSensitivity and
        converts it to a dict for use in pyOptSparse

        Parameters
        ----------
        dIdx : array
           Flattened array of length getNDV(). Generally it comes from
           a call to totalSensitivity()

        out1D : boolean
            If true, creates a 1D array in the dictionary instead of 2D.
            This function is used in the matrix-vector product calculation.

        Returns
        -------
        dIdxDict : dictionary
           Dictionary of the same information keyed by this object's
           design variables
        """

        ii = 0
        dIdxDict = {}
        for key in self.dvList:
            dv = self.dvDict[key]
            if out1D:
                dIdxDict[dv.name] = np.ravel(dIdx[:, ii : ii + dv.nVal])
            else:
                dIdxDict[dv.name] = dIdx[:, ii : ii + dv.nVal]
            ii += dv.nVal

        return dIdxDict

    def sens(self, dIdpt):
        # its important to remember that dIdpt are vector-like values,
        # so we don't apply the transformations and only the rotations!
        xdv = self.getValues()
        return self.funcCallback(dIdpt, mode="bwd", applyDisplacement=False, dvDict=xdv)

    def addVariablesPyOpt(
        self,
        optProb,
        ignoreVars=None,
        freezeVars=None,
    ):
        if ignoreVars is None:
            ignoreVars = set()
        if freezeVars is None:
            freezeVars = set()

        for key in self.dvList:
            if key not in ignoreVars:
                dv = self.dvDict[key]
                if key not in freezeVars:
                    optProb.addVarGroup(
                        dv.name,
                        dv.nVal,
                        "c",
                        value=dv.value.real,
                        lower=dv.lower,
                        upper=dv.upper,
                        scale=dv.scale,
                    )
                else:
                    optProb.addVarGroup(
                        dv.name,
                        dv.nVal,
                        "c",
                        value=dv.value.real,
                        lower=dv.value,
                        upper=dv.value,
                        scale=dv.scale,
                    )


class PointSet:
    def __init__(self, pointsBase, transformationFunc):
        self.pointsBase = pointsBase
        self.nPts = len(self.pointsBase)
        self.transformationFunc = transformationFunc
