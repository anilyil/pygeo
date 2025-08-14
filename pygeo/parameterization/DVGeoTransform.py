# Standard Python modules
from collections import OrderedDict
import copy

# External modules
from baseclasses.utils import Error
from mpi4py import MPI
import numpy as np

from .designVars import geoDVTransformation


class DVGeometryTransform:
    """
    This class is used to manage multiple components with their own Free-Form Deformation (FFD) parameterizations.
    This class acts as an intermediary between the top-level DVGeometry object and transformation functions.

    Parameters
    ----------
    DVGeoTop : DVGeometry
        The top-level DVGeometry object responsible for geometric design changes.
    name : str, optional
        A prefix for design variable names to ensure uniqueness in optimization problems.
        Defaults to None.
    checkDVs : bool, optional
        If True, checks for duplicate design variable names across components. Defaults to True.
    """

    def __init__(self, DVGeoTop, name=None, checkDVs=True):

        self.name = name
        self.DVGeoTop = DVGeoTop
        self.transormationFuncs = OrderedDict()
        self.points = OrderedDict()
        self.ptSetNames = []
        self.updated = {}
        self.checkDVs = checkDVs

    def addTransformationFunction(self, funcName, funcCallback):
        """
        Add a transformation function.

        This function allows users to define a custom transformation function
        that can be applied to point sets. The transformation function is stored
        in an internal dictionary for later use.

        Parameters
        ----------
        funcName : str
            The name of the transformation function. This name will be used to reference
            the function when applying transformations to point sets.
        funcCallback : callable
            A callable object (e.g., a function) that implements the transformation logic.
            The callback should accept points and other necessary parameters for the transformation.
        """

        # save the info in the dictionary
        self.transormationFuncs[funcName] = TransformationFunc(funcName, funcCallback)

    def addTransformationDV(
        self, funcName, dvName, value, lower=None, upper=None, scale=1.0, config=None, prependName=True
    ):
        """
        Add a design variable (DV) for a specific transformation function.

        This method associates a design variable with an existing transformation function.
        The design variable can then be used to control the parameters of the transformation.

        Parameters
        ----------
        funcName : str
            The name of the transformation function to which this design variable will be added.
            The function must have been previously added using `addTransformationFunction`.
        dvName : str
            The name of the design variable. This name must be unique across all DVs.
        value : float or array-like
            The initial value(s) of the design variable.
        lower : float or array-like, optional
            The lower bound(s) for the design variable. Defaults to None (no lower bound).
        upper : float or array-like, optional
            The upper bound(s) for the design variable. Defaults to None (no upper bound).
        scale : float, optional
            A scaling factor for the design variable. Defaults to 1.0.
        config : dict, optional
            Additional configuration options for the design variable. Defaults to None.
        prependName : bool, optional
            If True, prepends the parent DVGeometryTransform object's name to `dvName`
            for uniqueness. Defaults to True.

        Raises
        ------
        KeyError
            If `funcName` does not correspond to an existing transformation function.
        """

        if funcName not in self.transormationFuncs:
            raise Error(f"Transformation function '{funcName}' not found.")

        self.transormationFuncs[funcName].addDV(dvName, value, lower, upper, scale, config, prependName)

    def addPointSet(self, points, ptName, transformationFunc=None, config=None, **kwargs):
        """
        Add a set of points to be transformed and managed by the DVGeometryTransform object.

        This method allows users to add a set of points (e.g., coordinates) to the geometry.
        If a transformation function is specified, the points are transformed before being passed
        to the top-level DVGeometry object. Additionally, baseline design variables must be provided
        if a transformation function is used.

        Parameters
        ----------
        points : ndarray
            Array of points to be added. The input will be converted to at least 2D to handle single test points.
        ptName : str
            Name of the point set. This name is used to reference the point set in subsequent operations.
        transformationFunc : str or None, optional
            Name of the transformation function to apply. If None, no transformation is applied.
        config : dict or None, optional
            Configuration dictionary for additional settings.
        kwargs : dict
            Additional arguments for point set processing. Must include 'baselineDVs' if a transformation function is specified.

        Raises
        ------
        Error
            Raised if 'baselineDVs' is not provided in `kwargs` when using a transformation function.
        """

        # Ensure the input points are at least 2D in shape
        points = np.atleast_2d(points)

        # Apply the transformation function if specified
        if transformationFunc is not None:
            tfunc = self.transormationFuncs[transformationFunc]

            # Check for baseline design variables in kwargs
            if "baselineDVs" not in kwargs:
                raise Error(
                    "'baselineDVs' must be provided in the kwargs if this pointset uses a transformation function."
                )

            baselineDVs = kwargs.pop("baselineDVs")

            # Save the current design variable values
            DVRef = copy.deepcopy(tfunc.getValues())

            # Set baseline design variable values for the transformation function
            tfunc.setDesignVars(baselineDVs)

            # Transform points using the backward mode and baseline configuration
            pointsBase = tfunc.update(points, "bwd", config=config)

            # Restore the original design variable values
            tfunc.setDesignVars(DVRef)
        else:
            # If no transformation function is specified, use the original points directly
            pointsBase = points

        # Save point set information locally within this object
        self.points[ptName] = PointSet(pointsBase, transformationFunc)

        # Forward the point set to the top-level DVGeometry object for further management
        self.DVGeoTop.addPointSet(pointsBase, ptName, **kwargs)

        # Mark this point set as not updated initially
        self.updated[ptName] = False

    def setDesignVars(self, dvDict):
        """
        Set design variables from a dictionary.

        This method updates the design variables for all transformation functions and
        the top-level DVGeometry object using the provided dictionary. It also resets
        internal flags and Jacobians to ensure consistency after changes.

        Parameters
        ----------
        dvDict : dict
            Dictionary of design variables. The keys must correspond to the names of
            design variables defined in this object or its transformation functions.
            Any additional keys in the dictionary are ignored.

        Raises
        ------
        Error
            Raised if duplicate design variable names are detected across components
            when `checkDVs` is set to True.
        """
        # Check for duplicate design variable names if enabled
        if self.checkDVs:
            dvNames = self.getVarNames()
            duplicates = len(dvNames) != len(set(dvNames))
            if duplicates:
                raise Error(
                    "There are duplicate DV names in a component or across components. "
                    "If this is intended, initialize the DVGeometryTransform class with checkDVs=False."
                )

        # Update design variables for each transformation function
        for transFunc in self.transormationFuncs.values():
            transFunc.setDesignVars(dvDict)

        # Update design variables in the top-level DVGeometry object
        self.DVGeoTop.setDesignVars(dvDict)

        # Mark all point sets as not up-to-date after updating design variables
        for pointSet in self.updated:
            self.updated[pointSet] = False

        # Reset Jacobians for all point sets
        for ptSet in self.points.values():
            ptSet.dPtdDV = None

    def getValues(self):
        """
        Retrieve the current set of design variables.

        This method returns a dictionary containing the current values of all design variables
        managed by the DVGeometryTransform object. The dictionary includes variables from both
        the top-level DVGeometry object and all associated transformation functions. The output
        is suitable for use in a subsequent call to `setDesignVars`.

        Returns
        -------
        dvDict : dict
            A dictionary where keys are design variable names and values are their corresponding
            current values.
        """
        # Start by retrieving the design variables from the top-level DVGeometry object
        dvDict = self.DVGeoTop.getValues()

        # Update the dictionary with design variables from each transformation function
        for transFunc in self.transormationFuncs.values():
            dvDict.update(transFunc.getValues())

        return dvDict

    def update(self, ptSetName, config=None):
        """
        Update the coordinates of a specified point set based on the current design variables.

        This method retrieves the point set identified by `ptSetName` from the top-level DVGeometry object,
        applies any associated transformation functions, and marks the point set as up-to-date.

        Parameters
        ----------
        ptSetName : str
            The name of the point set to update. This must match one of the names provided in a prior
            call to `addPointSet`.
        config : dict or None, optional
            Configuration dictionary for additional settings. Defaults to None.

        Returns
        -------
        points : ndarray
            The updated coordinates of the specified point set after applying all transformations.
        """
        # Retrieve updated points from the top-level DVGeometry object
        points = self.DVGeoTop.update(ptSetName, config=config)

        # Check if a transformation function is associated with this point set
        transFunc = self.points[ptSetName].transformationFunc
        if transFunc is not None:
            # Apply the transformation function in forward mode
            points = self.transormationFuncs[transFunc].update(points, "fwd", config=config)

        # Mark this point set as up-to-date
        self.updated[ptSetName] = True

        return points

    def pointSetUpToDate(self, ptSetName):
        """
        Check if a specified point set is up-to-date.

        This method is used to query whether the coordinates of a point set have been
        updated to reflect the current design variable values. When `update()` is called
        for a point set, its `updated` flag is set to True. If design variables are reset
        using `setDesignVars`, all flags are reset to False, as no point sets are considered
        up-to-date anymore.

        Parameters
        ----------
        ptSetName : str
            The name of the point set to check. This must match one of the names provided
            in a prior call to `addPointSet`.

        Returns
        -------
        bool
            True if the point set is up-to-date, False otherwise.
            If the point set does not exist, it returns True by default.
        """
        # Check if the point set exists in the updated dictionary
        if ptSetName in self.updated:
            return self.updated[ptSetName]
        else:
            # If the point set does not exist, assume it is up-to-date by default
            return True

    def getNDV(self):
        """
        Calculate and return the total number of design variables.

        This method computes the total number of design variables managed by the DVGeometryTransform object.
        It includes design variables from both the top-level DVGeometry object and all associated transformation functions.

        Returns
        -------
        nDV : int
            The total number of design variables across all components and transformation functions.
        """
        # Start with the number of design variables in the top-level DVGeometry object
        nDV = self.DVGeoTop.getNDV()

        # Add the number of design variables from each transformation function
        for transFunc in self.transormationFuncs.values():
            nDV += transFunc.getNDV()

        return nDV

    def getVarNames(self, **kwargs):
        """
        Retrieve a list of design variable names.

        This method returns the names of all design variables managed by the DVGeometryTransform object.
        It includes variables from both the top-level DVGeometry object and all associated transformation functions.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments forwarded to the top-level DVGeometry object's `getVarNames` method.
            For example, `pyOptSparse=True` can be passed to format variable names for pyOptSparse.

        Returns
        -------
        dvNames : list of str
            A list of design variable names.

        Examples
        --------
        >>> optProb.addCon(..., wrt=DVGeo.getVarNames(pyOptSparse=True))
        """
        # Retrieve variable names from the top-level DVGeometry object, passing kwargs
        dvNames = self.DVGeoTop.getVarNames(**kwargs)

        # Append variable names from each transformation function
        for transFunc in self.transormationFuncs.values():
            dvNames.extend(transFunc.getVarNames())

        return dvNames

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        """
        Compute the total sensitivity of functions of interest with respect to design variables.

        This method calculates the total derivative (\( \frac{\partial I}{\partial x} \)) of one or more
        functions of interest (e.g., aerodynamic forces or performance metrics) with respect to all design
        variables. The input `dIdpt` represents the Jacobian matrix of the functions with respect to the
        point coordinates.

        Parameters
        ----------
        dIdpt : ndarray
            Total Jacobian matrix of the functions of interest with respect to point coordinates.
            Shape: (N, M, 3), where:
            - N: Number of functions of interest.
            - M: Number of points.
            - 3: x-y-z dimensions.
        ptSetName : str
            Name of the point set for which sensitivities are being computed. Must match a previously added point set.
        comm : MPI.Comm or None, optional
            MPI communicator for parallel reduction. If None, no parallel reduction is performed.
        config : dict or None, optional
            Configuration dictionary for additional settings. Defaults to None.

        Returns
        -------
        dIdxDict : dict
            Dictionary mapping design variable names to their sensitivities. Each entry contains an array
            of shape (N,) or (N, k), where k is the number of values for that design variable.

        Notes
        -----
        - The method first processes local derivatives using transformation functions (if applicable).
        - It then computes sensitivities for the top-level DVGeometry object and combines them into a single dictionary.
        """
        # Ensure dIdpt has at least 3 dimensions
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])

        N = dIdpt.shape[0]  # Number of functions of interest
        dIdxDict = {}

        # Process local derivatives using transformation functions if applicable
        transFuncName = self.points[ptSetName].transformationFunc
        if transFuncName is not None:
            transFunc = self.transormationFuncs[transFuncName]
            pointsBase = self.points[ptSetName].pointsBase

            # Compute or retrieve the Jacobian for this point set
            if self.points[ptSetName].dPtdDV is None:
                dPtdDV = transFunc.getJacobian(pointsBase, config=config)
                self.points[ptSetName].dPtdDV = dPtdDV  # Cache the Jacobian for reuse
            else:
                dPtdDV = self.points[ptSetName].dPtdDV

            # Compute local sensitivities by contracting dIdpt with dPtdDV
            dIdx_local = np.tensordot(dIdpt, dPtdDV, axes=([1, 2], [1, 2]))

            # Perform parallel reduction if MPI communicator is provided
            if comm:
                dIdxArray = comm.allreduce(dIdx_local, op=MPI.SUM)
            else:
                dIdxArray = dIdx_local

            # Convert local sensitivities to a dictionary format
            dIdxDict.update(transFunc.convertSensitivityToDict(dIdxArray))

            # Rotate dIdpt into the DVGeometry reference frame for further processing
            for ifunc in range(N):
                dIdpt[ifunc] = transFunc.sens(dIdpt[ifunc])

        # Process sensitivities for the top-level DVGeometry object and combine results
        dIdxDict.update(self.DVGeoTop.totalSensitivity(dIdpt, ptSetName, comm=comm, config=config))

        return dIdxDict

    def addVariablesPyOpt(self, optProb, ignoreVars=None, freezeVars=None, **kwargs):
        """
        Add the current set of design variables to the pyOptSparse optimization problem.

        This method integrates the design variables managed by both the top-level DVGeometry object
        and transformation functions into the provided pyOptSparse optimization problem (`optProb`).
        Top-level DVGeometry-specific options are passed as keyword arguments (`kwargs`).

        Parameters
        ----------
        optProb : pyOpt.Optimization
            The optimization problem definition to which design variables are added.
        ignoreVars : list of str, optional
            List of design variable names to exclude from the optimization problem. Defaults to None.
        freezeVars : list of str, optional
            List of design variable names to add as optimization variables but with their bounds fixed
            at their current values. Defaults to None.
        kwargs : dict
            Additional keyword arguments forwarded to the top-level DVGeometry object's `addVariablesPyOpt` method.
            These may include options such as `globalVars`, `localVars`, `sectionlocalVars`, and `comps`.

        Notes
        -----
        - Design variables from the top-level DVGeometry object are added first using the provided `kwargs`.
        - Transformation function-specific design variables are added afterward.
        - Frozen variables are included in the optimization problem but have fixed bounds.

        Examples
        --------
        >>> DVGeoTransform.addVariablesPyOpt(optProb, ignoreVars=['dv1'], freezeVars=['dv2'], globalVars=True)
        """
        # Add design variables from the top-level DVGeometry object using kwargs
        self.DVGeoTop.addVariablesPyOpt(optProb, ignoreVars=ignoreVars, freezeVars=freezeVars, **kwargs)

        # Add transformation function-specific design variables
        for transformFunc in self.transormationFuncs.values():
            transformFunc.addVariablesPyOpt(optProb, ignoreVars=ignoreVars, freezeVars=freezeVars)

    def getLocalIndex(self, iVol):
        """
        Return the local index mapping that points to the global coefficient list for a given FFD volume.

        This method queries the top-level DVGeometry object to retrieve the local-to-global index mapping
        for a specified Free-Form Deformation (FFD) volume.

        Parameters
        ----------
        iVol : int
            Index specifying the FFD volume for which the local index mapping is requested.

        Returns
        -------
        ndarray
            A copy of the local index mapping array corresponding to the specified FFD volume.
        """
        # Retrieve the local index mapping from the top-level DVGeometry object
        return self.DVGeoTop.getLocalIndex(iVol)


class TransformationFunc:
    """
    A class to manage geometric transformations and their associated design variables.

    This class represents a single transformation function applied to point sets. It stores
    the design variables associated with the transformation, provides methods for updating
    point coordinates based on design variables, and computes sensitivities for optimization.

    Parameters
    ----------
    name : str
        Name of the transformation function. Used as a prefix for design variable names.
    funcCallback : callable
        A function or callable object that implements the transformation logic. It should accept
        points, mode, configuration, and design variable dictionary as inputs.

    Attributes
    ----------
    dvDict : OrderedDict
        Dictionary of design variables associated with this transformation function.
        Keys are design variable names, and values are `geoDVTransformation` objects.
    dvList : list of str
        List of design variable names in the order they were added.
    funcCallback : callable
        The transformation function callback provided during initialization.
    name : str
        Name of this transformation function.
    isComplex : bool
        Flag indicating whether complex-step derivatives are being used.
    """

    def __init__(self, name, funcCallback):
        self.dvDict = OrderedDict()  # Dictionary to store design variables
        self.dvList = []  # List of design variable names
        self.funcCallback = funcCallback  # Transformation callback function
        self.name = name  # Name of the transformation function
        self.isComplex = False  # Flag for complex-step derivatives

    def addDV(self, dvName, value, lower, upper, scale, config, prependName):
        """
        Add a design variable to this transformation function.

        Parameters
        ----------
        dvName : str
            Name of the design variable.
        value : float or array-like
            Initial value(s) of the design variable.
        lower : float or array-like
            Lower bound(s) for the design variable.
        upper : float or array-like
            Upper bound(s) for the design variable.
        scale : float
            Scaling factor for the design variable.
        config : dict
            Configuration dictionary for additional settings.
        prependName : bool
            If True, prepends the parent name to `dvName` for uniqueness.
        """
        if self.name is not None and prependName:
            dvName = f"{self.name}_{dvName}"

        self.dvDict[dvName] = geoDVTransformation(dvName, value, lower, upper, scale, config)
        self.dvList.append(dvName)

    def setDesignVars(self, dvDict):
        """
        Update the values of design variables for this transformation function.

        This method updates the internal design variable values based on the input dictionary `dvDict`.
        It ensures that the number of values provided matches the expected number for each design variable.

        Parameters
        ----------
        dvDict : dict
            Dictionary containing new values for the design variables. The keys must correspond to
            existing design variable names in this transformation function.

        Raises
        ------
        Error
            Raised if the number of values provided for a design variable does not match the expected number.
        """

        def _checkArrLength(key, nIn, nRef):
            """
            Helper function to check if the length of input values matches the expected length.

            Parameters
            ----------
            key : str
                Name of the design variable being checked.
            nIn : int
                Number of input values provided.
            nRef : int
                Expected number of values for this design variable.

            Raises
            ------
            Error
                Raised if `nIn` does not match `nRef`.
            """
            if nIn != nRef:
                raise Error(
                    f"Incorrect number of design variables for DV: {key}.\n"
                    f"Expecting {nRef} variables but received {nIn}"
                )

        # Update each design variable in this transformation function
        for key in dvDict:
            if key in self.dvDict:
                # Ensure input values are at least 1D
                vals_to_set = np.atleast_1d(dvDict[key])

                # Check that the number of input values matches expectations
                _checkArrLength(key, len(vals_to_set), self.dvDict[key].nVal)

                # Update the value of the design variable
                self.dvDict[key].value = vals_to_set

    def getValues(self):
        """
        Retrieve the current set of design variables for this transformation function.

        This method returns a dictionary containing the current values of all design variables
        associated with this transformation function. The output is suitable for use in a subsequent
        call to `setDesignVars`.

        Returns
        -------
        dvDict : dict
            Dictionary of design variables. Keys are the names of the design variables, and values
            are their corresponding current values.

        Notes
        -----
        - The returned dictionary includes all design variables in the order they were added.
        - This method is typically used during optimization or sensitivity analysis to retrieve
        the current state of the design variables.
        """
        # Initialize an empty dictionary to store design variable values
        dvDict = {}

        # Loop through all design variable names and retrieve their values
        for key in self.dvList:
            dvDict[key] = self.dvDict[key].value

        return dvDict

    def update(self, points, mode, config=None):
        """
        Apply the transformation function to a set of points.

        This method updates the coordinates of the input points based on the current design
        variables and the specified mode. The transformation logic is implemented via the
        callback function provided during initialization.

        Parameters
        ----------
        points : ndarray
            Array of point coordinates to be transformed.
        mode : str
            Mode of transformation. Typically "fwd" for forward transformations or "bwd" for backward transformations.
        config : dict or None, optional
            Configuration dictionary for additional settings during transformation. Defaults to None.

        Returns
        -------
        transformedPoints : ndarray
            Array of transformed point coordinates after applying the transformation function.

        Notes
        -----
        - The transformation function callback is called with the current design variable values (`dvDict`).
        - The `applyDisplacement` flag is set to True for forward transformations and False for backward transformations.
        """
        # Call the transformation function callback with current design variables and configuration
        return self.funcCallback(points, mode=mode, applyDisplacement=True, config=config, dvDict=self.getValues())

    def getNDV(self):
        """
        Calculate and return the total number of design variables associated with this transformation function.

        This method computes the total number of design variables managed by this transformation function,
        summing up the number of values for each design variable.

        Returns
        -------
        nDV : int
            Total number of design variables across all variables in this transformation function.

        Notes
        -----
        - Each design variable may have multiple values (e.g., array-like variables), and these are included in the total count.
        """
        # Initialize counter for total number of design variables
        nDV = 0

        # Sum up the number of values for each design variable in this transformation function
        for dvName in self.dvList:
            nDV += self.dvDict[dvName].nVal

        return nDV

    def getVarNames(self):
        """
        Retrieve the names of all design variables associated with this transformation function.

        This method returns a list containing the names of all design variables managed by this
        transformation function. The names are stored in the order they were added.

        Returns
        -------
        dvList : list of str
            A list of design variable names.

        Notes
        -----
        - This method is typically used when specifying optimization constraints or objectives.
        - The returned list directly corresponds to the internal storage of design variable names.

        Examples
        --------
        >>> varNames = transformFunc.getVarNames()
        >>> print(varNames)
        ['rotation_angle', 'translation_x', 'scaling_factor']
        """
        # Return the list of design variable names
        return self.dvList

    def getJacobian(self, pointsBase, config=None):
        """
        Compute the Jacobian matrix of the point coordinates with respect to the design variables.

        This method calculates the sensitivity of each point coordinate in `pointsBase` with respect
        to all design variables managed by this transformation function. The Jacobian is computed
        using complex-step differentiation for accuracy.

        Parameters
        ----------
        pointsBase : ndarray
            Array of baseline point coordinates to compute the Jacobian for.
            Shape: (nPts, 3), where nPts is the number of points and 3 corresponds to x-y-z dimensions.
        config : dict or None, optional
            Configuration dictionary for additional settings during Jacobian computation. Defaults to None.

        Returns
        -------
        dPtdDV : ndarray
            The Jacobian matrix of shape (nDV, nPts, 3), where:
            - nDV: Total number of design variables.
            - nPts: Number of points.
            - 3: x-y-z dimensions.

        Notes
        -----
        - Complex-step differentiation is used for high-accuracy derivative computation.
        - The Jacobian is computed by perturbing each design variable individually and measuring
        the resulting change in the transformed points.
        """
        dh = 1e-40  # Step size for complex-step differentiation

        # Create a deep copy of the current design variable values and cast them to complex type
        xdvCmplx = copy.deepcopy(self.getValues())
        for key, val in xdvCmplx.items():
            xdvCmplx[key] = val.astype("complex")

        # Allocate space for the Jacobian matrix
        nPts = pointsBase.shape[0]
        dPtdDV = np.zeros((self.getNDV(), nPts, 3), dtype=float)

        # Loop over all design variables and compute partial derivatives
        ii = 0
        for key in self.dvList:
            dv = self.dvDict[key]
            for jj in range(dv.nVal):
                # Save the reference value and perturb it with a complex step
                refVal = xdvCmplx[key][jj]
                xdvCmplx[key][jj] = refVal + dh * 1j

                # Evaluate the transformation function with perturbed design variables
                pointsPlus = self.funcCallback(
                    pointsBase,
                    mode="fwd",
                    applyDisplacement=True,
                    config=config,
                    dvDict=xdvCmplx,
                )

                # Compute the derivative using the imaginary part (complex-step differentiation)
                dPtdDV[ii + jj] = np.imag(pointsPlus) / dh

                # Reset the perturbed design variable to its original value
                xdvCmplx[key][jj] = refVal

            ii += dv.nVal

        return dPtdDV

    def convertSensitivityToDict(self, dIdx, out1D=False):
        """
        Convert sensitivity array to a dictionary format for use in pyOptSparse.

        This method takes the result of a sensitivity calculation (e.g., from `totalSensitivity`)
        and converts it into a dictionary format. Each entry in the dictionary corresponds to
        a design variable managed by this transformation function.

        Parameters
        ----------
        dIdx : ndarray
            Flattened array of sensitivities with shape (nFuncs, nDV), where:
            - nFuncs: Number of functions of interest.
            - nDV: Total number of design variables.
            Typically obtained from a call to `totalSensitivity()`.
        out1D : bool, optional
            If True, creates 1D arrays in the dictionary instead of 2D arrays. Defaults to False.
            This option is typically used during matrix-vector product calculations.

        Returns
        -------
        dIdxDict : dict
            Dictionary containing sensitivities keyed by design variable names. Each entry is either:
            - A 2D array with shape (nFuncs, nVal) if `out1D=False`.
            - A 1D array with shape (nFuncs * nVal,) if `out1D=True`.

        Notes
        -----
        - The method iterates over all design variables managed by this transformation function.
        - Sensitivities for each design variable are extracted from the input array and stored in the dictionary.
        """
        ii = 0  # Initialize index for slicing sensitivity array
        dIdxDict = {}  # Initialize output dictionary

        # Iterate over all design variables in this transformation function
        for key in self.dvList:
            dv = self.dvDict[key]

            # Extract sensitivities for this design variable
            if out1D:
                # Flatten the sensitivity values into a 1D array
                dIdxDict[dv.name] = np.ravel(dIdx[:, ii : ii + dv.nVal])
            else:
                # Keep the sensitivity values as a 2D array
                dIdxDict[dv.name] = dIdx[:, ii : ii + dv.nVal]

            # Increment index for next design variable
            ii += dv.nVal

        return dIdxDict

    def sens(self, dIdpt):
        """
        Apply sensitivity transformations to a given derivative vector.

        This method computes the sensitivity of functions of interest (\( \frac{\partial I}{\partial x} \))
        with respect to design variables by applying backward transformations. Unlike forward transformations,
        this method only applies rotations to the input derivative vector, as displacement transformations
        are not relevant for sensitivities.

        Parameters
        ----------
        dIdpt : ndarray
            Array representing the derivative of functions of interest with respect to point coordinates.
            Shape: (nPts, 3), where:
            - nPts: Number of points.
            - 3: x-y-z dimensions.

        Returns
        -------
        transformedSens : ndarray
            Array of transformed sensitivities after applying backward transformations.
            Shape matches the input `dIdpt`.

        Notes
        -----
        - Sensitivity transformations are applied using the transformation function callback.
        - Displacement transformations are disabled (`applyDisplacement=False`) since they are not required for sensitivities.
        """
        # Retrieve current design variable values
        xdv = self.getValues()

        # Apply backward transformation to the sensitivity vector using the callback function
        return self.funcCallback(dIdpt, mode="bwd", applyDisplacement=False, dvDict=xdv)

    def addVariablesPyOpt(
        self,
        optProb,
        ignoreVars=None,
        freezeVars=None,
    ):
        """
        Add design variables associated with this transformation function to a pyOptSparse optimization problem.

        This method integrates the design variables managed by this transformation function into the provided
        pyOptSparse optimization problem (`optProb`). It supports ignoring specific variables or freezing them
        by setting their bounds to their current values.

        Parameters
        ----------
        optProb : pyOpt.Optimization
            The optimization problem definition to which design variables are added.
        ignoreVars : set of str, optional
            Set of design variable names to exclude from the optimization problem. Defaults to an empty set.
        freezeVars : set of str, optional
            Set of design variable names to add as optimization variables but with their bounds fixed at their
            current values. Defaults to an empty set.

        Notes
        -----
        - Design variables that are in `ignoreVars` are excluded entirely from the optimization problem.
        - Design variables that are in `freezeVars` are added with their lower and upper bounds set to their
        current values, effectively freezing them.

        Examples
        --------
        >>> transformFunc.addVariablesPyOpt(optProb, ignoreVars={"rotation_angle"}, freezeVars={"scaling_factor"})
        """
        # Initialize empty sets if ignoreVars or freezeVars are not provided
        if ignoreVars is None:
            ignoreVars = set()
        if freezeVars is None:
            freezeVars = set()

        # Iterate over all design variables in this transformation function
        for key in self.dvList:
            if key not in ignoreVars:  # Skip ignored variables
                dv = self.dvDict[key]
                if key not in freezeVars:
                    # Add variable as a standard optimization variable
                    optProb.addVarGroup(
                        dv.name,
                        dv.nVal,
                        "c",  # Continuous variable type
                        value=dv.value.real,
                        lower=dv.lower,
                        upper=dv.upper,
                        scale=dv.scale,
                    )
                else:
                    # Add variable as a frozen optimization variable with fixed bounds
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
    """
    A class to represent a set of points in the geometry and manage their transformations.

    This class stores the baseline coordinates of a point set and manages its association
    with a transformation function. It also provides storage for Jacobian matrices used
    in sensitivity analysis.

    Parameters
    ----------
    pointsBase : ndarray
        Array of baseline point coordinates for this point set.
        Shape: (nPts, 3), where:
        - nPts: Number of points.
        - 3: x-y-z dimensions.
    transformationFunc : str or None
        Name of the transformation function associated with this point set.
        If None, no transformation function is applied.

    Attributes
    ----------
    pointsBase : ndarray
        Baseline coordinates of the points in this point set.
    nPts : int
        Number of points in this point set.
    transformationFunc : str or None
        Name of the associated transformation function, if any.
    dPtdDV : ndarray or None
        Jacobian matrix of point coordinates with respect to design variables.
        Shape: (nDV, nPts, 3), where:
        - nDV: Total number of design variables.
        - nPts: Number of points.
        - 3: x-y-z dimensions.
        Defaults to None until computed during sensitivity analysis.
    """

    def __init__(self, pointsBase, transformationFunc):
        # Store the baseline coordinates of the points
        self.pointsBase = pointsBase

        # Number of points in the point set
        self.nPts = len(self.pointsBase)

        # Name of the associated transformation function (if any)
        self.transformationFunc = transformationFunc

        # Jacobian matrix (initialized as None)
        self.dPtdDV = None
