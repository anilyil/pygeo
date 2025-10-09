# Standard Python modules
from collections import OrderedDict

# External modules
from baseclasses.utils import Error
from mpi4py import MPI
import numpy as np
from scipy import sparse

from pygeo.geo_utils import readNValues
try:
    # External modules
    from pysurf import (
        adtAPI,
        adtAPI_cs,
        tsurf_tools,
    )

    pysurfInstalled = True
except ImportError:
    pysurfInstalled = False

from .CompIntersection import ComponentIntersection
from .WarpIntersection import WarpedIntersection


class DVGeometryMulti:
    """
    A class for manipulating multiple components using multiple FFDs
    and handling design changes near component intersections.

    Parameters
    ----------
    comm : MPI.IntraComm, optional
       The communicator associated with this geometry object.
       This is also used to parallelize the triangulated meshes.

    checkDVs : bool, optional
        Flag to check whether there are duplicate DV names in or across components.

    debug : bool, optional
        Flag to generate output useful for debugging the intersection setup.

    isComplex : bool, optional
        Flag to use complex variables for complex step verification.

    """

    def __init__(self, comm=MPI.COMM_WORLD, checkDVs=True, debug=False, isComplex=False):
        # Check to make sure pySurf is installed before initializing
        if not pysurfInstalled:
            raise ImportError("pySurf is not installed and is required to use DVGeometryMulti.")

        self.compNames = []
        self.comps = OrderedDict()
        self.DVGeoDict = OrderedDict()
        self.points = OrderedDict()
        self.comm = comm
        self.updated = {}
        self.intersectComps = []
        self.checkDVs = checkDVs
        self.debug = debug
        self.complex = isComplex
        self.curConfig = None
        self.updateICs = True

        # Set real or complex Fortran API
        if isComplex:
            self.dtype = complex
            self.adtAPI = adtAPI_cs.adtapi
        else:
            self.dtype = float
            self.adtAPI = adtAPI.adtapi

    def addComponent(self, comp, DVGeo, triMesh=None, scale=1.0, bbox=None, pointSetKwargs=None, supportCurves=None):
        """
        Method to add components to the DVGeometryMulti object.

        Parameters
        ----------
        comp : str
            The name of the component.

        DVGeo : DVGeometry
            The DVGeometry object defining the component FFD.

        triMesh : str, optional
            Path to the triangulated mesh file for this component.

        scale : float, optional
            A multiplicative scaling factor applied to the triangulated mesh coordinates.
            Useful for when the scales of the triangulated and CFD meshes do not match.

        bbox : dict, optional
            Specify a bounding box that is different from the bounds of the FFD.
            The keys can include ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``.
            If any of these are not provided, the FFD bound is used.

        pointSetKwargs : dict, optional
            Keyword arguments to be passed to the component addPointSet call for the triangulated mesh.

        """

        # Assign mutable defaults
        if bbox is None:
            bbox = {}
        if pointSetKwargs is None:
            pointSetKwargs = {}
        if supportCurves is None:
            supportCurves = {}

        if triMesh is not None:
            # We also need to read the triMesh and save the points
            nodes, triConn, triConnStack, barsConn = self._readCGNSFile(triMesh)

            # scale the nodes
            nodes *= scale

            # also read the support curves if we have any
            for scName, scData in supportCurves.items():
                scNodes = scData["nodes"]
                scConn = scData["conn"]

                # increment the conn by this number
                nNode = len(nodes)
                scConn += nNode
                # save the conn array
                barsConn[scName] = scConn
                # append the nodes to the node array
                nodes = np.vstack([nodes, scNodes])

            # We will split up the points by processor when adding them to the component DVGeo

            # Compute the processor sizes with integer division
            sizes = np.zeros(self.comm.size, dtype="intc")
            nPts = nodes.shape[0]
            sizes[:] = nPts // self.comm.size

            # Add the leftovers
            sizes[: nPts % self.comm.size] += 1

            # Compute the processor displacements
            disp = np.zeros(self.comm.size + 1, dtype="intc")
            disp[1:] = np.cumsum(sizes)

            # Save the size and displacement in a dictionary
            triMeshData = {}
            triMeshData["sizes"] = sizes
            triMeshData["disp"] = disp

            # Split up the points into the points for this processor
            procNodes = nodes[disp[self.comm.rank] : disp[self.comm.rank + 1]]

            # Add these points to the component DVGeo
            DVGeo.addPointSet(procNodes, f"triMesh_{comp}", **pointSetKwargs)
        else:
            # the user has not provided a triangulated surface mesh for this file
            nodes = None
            triConn = None
            triConnStack = None
            barsConn = None
            triMeshData = None

        # we will need the bounding box information later on, so save this here
        xMin, xMax = DVGeo.FFD.getBounds()

        # also we might want to modify the bounding box if the user specified any coordinates
        if "xmin" in bbox:
            xMin[0] = bbox["xmin"]
        if "ymin" in bbox:
            xMin[1] = bbox["ymin"]
        if "zmin" in bbox:
            xMin[2] = bbox["zmin"]
        if "xmax" in bbox:
            xMax[0] = bbox["xmax"]
        if "ymax" in bbox:
            xMax[1] = bbox["ymax"]
        if "zmax" in bbox:
            xMax[2] = bbox["zmax"]

        # initialize the component object
        self.comps[comp] = component(comp, DVGeo, nodes, triConn, triConnStack, barsConn, xMin, xMax, triMeshData, pointSetKwargs)

        # add the name to the list
        self.compNames.append(comp)

        # also save the DVGeometry pointer in the dictionary we pass back
        self.DVGeoDict[comp] = DVGeo

    def addIntersection(
        self,
        compA,
        compB,
        dStarA=0.2,
        dStarB=0.2,
        featureCurves=None,
        distTol=1e-14,
        project=False,
        marchDir=1,
        includeCurves=False,
        slidingCurves=None,
        intDir=None,
        curveEpsDict=None,
        trackSurfaces=None,
        excludeSurfaces=None,
        remeshBwd=True,
        anisotropy=[1.0, 1.0, 1.0],
        blendOrder=3,
        intersectionCurves=None,
        eps=1e-30,
        supportCurves=None,  # TODO Exclude from projections
    ):
        """
        Method that defines intersections between components.

        Parameters
        ----------
        compA : str
            The name of the first component.

        compB : str
            The name of the second component.

        dStarA : float, optional
            Distance from the intersection over which the inverse-distance deformation is applied on compA.

        dStarB : float, optional
            Distance from the intersection over which the inverse-distance deformation is applied on compB.

        featureCurves : list or dict, optional
            Points on feature curves will remain on the same curve after deformations and projections.
            Feature curves can be specified as a list of curve names.
            In this case, the march direction for all curves is ``marchDir``.
            Alternatively, a dictionary can be provided.
            In this case, the keys are the curve names and the values are the march directions for each curve.
            See ``marchDir`` for the definition of march direction.

        distTol : float, optional
            Distance tolerance to merge nearby nodes in the intersection curve.

        project : bool, optional
            Flag to specify whether to project points to curves and surfaces after the deformation step.

        marchDir : int, optional
            The side of the intersection where the feature curves are remeshed.
            The sign determines the direction and the value (1, 2, 3) specifies the axis (x, y, z).
            If ``remeshBwd`` is True, the other side is also remeshed.
            In this case, the march direction only serves to define the 'free end' of the feature curve.
            If None, the entire curve is remeshed.
            This argument is only used if a list is provided for ``featureCurves``.

        includeCurves : bool, optional
            Flag to specify whether to include features curves in the inverse-distance deformation.

        slidingCurves : list, optional
            The list of curves to project to, but on which the mesh nodes are not frozen in their initial positions.
            This allows the mesh nodes to slide along the feature curve.

        intDir : int, optional
            If there are multiple intersection curves, this specifies which curve to choose.
            The sign determines the direction and the value (1, 2, 3) specifies the axis (x, y, z).
            For example, -1 specifies the intersection curve as the one that is further in the negative x-direction.

        curveEpsDict : dict, optional
            Required if using feature curves.
            The keys of the dictionary are the curve names and the values are distances.
            All points within the specified distance from the curve are considered to be on the curve.

        trackSurfaces : dict, optional
            Points on tracked surfaces will remain on the same surfaces after deformations and projections.
            The keys of the dictionary are the surface names and the values are distances.
            All points within the specified distance from the surface are considered to be on the surface.

        excludeSurfaces : dict, optional
            Points on excluded surfaces are removed from the intersection computations.
            The keys of the dictionary are the surface names and the values are distances.
            All points within the specified distance from the surface are considered to be on the surface.

        remeshBwd : bool, optional
            Flag to specify whether to remesh feature curves on the side opposite that
            which is specified by the march direction.

        anisotropy : list of float, optional
            List with three entries specifying scaling factors in the [x, y, z] directions.
            The factors multiply the [x, y, z] distances used in the curve-based deformation.
            Smaller factors in a certain direction will amplify the effect of the parts of the curve
            that lie in that direction from the points being warped.
            This tends to increase the mesh quality in one direction at the expense of other directions.
            This can be useful when the initial intersection curve is skewed.

        blendOrder : float, optional
            Power to use in the blending function.
            Default is 3, which will result in a cubic blend function.
            Typical values are between 1 and 3.
            This option controls how the intersection changes are blended into the rest of the shape.
            A higher blend order will preserve the mesh angles near the intersection better, but will provide a smaller range of motion.
            Linear blend (order 1) will provide the maximum range of motion, but the mesh will get skewed near the intersection.
            A unique benefit of the cubic blend is that it will maintain G2 continuity of the affected area and its boundaries.
            # TODO check if this last statement is true. maybe we can modify the functions so that any order higher than 2-3 results in continuous curvature

        intersectionCurves : list, optional
            List of feature curves that we use to track the intersection topology.
            If this is not provided, we use all of the curves on compB that also have a "marchDir.
            If this list is provided, we internally use this set of curves when determining the intersection topology.

        """

        # Assign mutable defaults
        if featureCurves is None:
            featureCurves = []
        if slidingCurves is None:
            slidingCurves = []
        if curveEpsDict is None:
            curveEpsDict = {}
        if trackSurfaces is None:
            trackSurfaces = {}
        if excludeSurfaces is None:
            excludeSurfaces = {}
        if intersectionCurves is None:
            # TODO this breaks if its not provided
            intersectionCurves = {}
        if supportCurves is None:
            supportCurves = {}

        nIC = len(self.intersectComps)

        # just initialize the intersection object
        self.intersectComps.append(
            ComponentIntersection(
                compA,
                compB,
                dStarA,
                dStarB,
                featureCurves,
                distTol,
                self,
                project,
                marchDir,
                includeCurves,
                slidingCurves,
                supportCurves,
                intDir,
                curveEpsDict,
                trackSurfaces,
                excludeSurfaces,
                remeshBwd,
                intersectionCurves,
                anisotropy,
                blendOrder,
                self.debug,
                self.dtype,
                nIC,
                eps,
            )
        )

    def addWarpedIntersection(
        self,
        compA,
        compB,
        intersectionFile,
        dStarA=0.2,
        dStarB=0.2,
        anisotropy=[1.0, 1.0, 1.0],
        blendOrder=3,
    ):
        """
        Method that defines intersections between components.

        Parameters
        ----------
        compA : str
            The name of the first component.

        compB : str
            The name of the second component.

        intersectionFile : str
            Path for the file defining the intersection curve.
            Must be in Plot3D format.

        dStarA : float, optional
            Distance from the intersection over which the inverse-distance deformation is applied on compA. If one of the dStar values is set to None, the algorithm is slightly modified. In this case, we let the component with a dStar of None to maintain the changes from its own FFD, while the other component observes a decay in its own FFD changes leading up to the intersection. The intersection purely follows the component with dStar of None in this case.

        dStarB : float, optional
            Distance from the intersection over which the inverse-distance deformation is applied on compB.

        anisotropy : list of float, optional
            List with three entries specifying scaling factors in the [x, y, z] directions.
            The factors multiply the [x, y, z] distances used in the curve-based deformation.
            Smaller factors in a certain direction will amplify the effect of the parts of the curve
            that lie in that direction from the points being warped.
            This tends to increase the mesh quality in one direction at the expense of other directions.
            This can be useful when the initial intersection curve is skewed.

        blendOrder : float, optional
            Power to use in the blending function.
            Default is 3, which will result in a cubic blend function.
            Typical values are between 1 and 3.
            This option controls how the intersection changes are blended into the rest of the shape.
            A higher blend order will preserve the mesh angles near the intersection better, but will provide a smaller range of motion.
            Linear blend (order 1) will provide the maximum range of motion, but the mesh will get skewed near the intersection.
            A unique benefit of the cubic blend is that it will maintain G2 continuity of the affected area and its boundaries.
            # TODO check if this last statement is true. maybe we can modify the functions so that any order higher than 2-3 results in continuous curvature

        """

        # read the plot3d curve
        nodes, barsConn = self._readP3DCurve(intersectionFile)

        nIC = len(self.intersectComps)

        # initialize the intersection object
        self.intersectComps.append(
            WarpedIntersection(
                compA,
                compB,
                dStarA,
                dStarB,
                nodes,
                barsConn,
                self,
                anisotropy,
                blendOrder,
                self.debug,
                self.dtype,
                nIC,
            )
        )

    def getDVGeoDict(self):
        """Return a dictionary of component DVGeo objects."""
        return self.DVGeoDict

    def addPointSet(self, points, ptName, compNames=None, comm=None, applyIC=False, **kwargs):
        """
        Add a set of coordinates to DVGeometryMulti.
        The is the main way that geometry, in the form of a coordinate list, is manipulated.

        Parameters
        ----------
        points : array, size (N,3)
            The coordinates to embed.
            These coordinates should all be inside at least one FFD volume.
        ptName : str
            A user supplied name to associate with the set of coordinates.
            This name will need to be provided when updating the coordinates
            or when getting the derivatives of the coordinates.
        compNames : list, optional
            A list of component names that this point set should be added to.
            To ease bookkeepping, an empty point set with ptName will be added to components not in this list.
            If a list is not provided, this point set is added to all components.
        comm : MPI.IntraComm, optional
            The communicator that is associated with the added point set.
        applyIC : bool, optional
            Flag to specify whether this point set will follow the updated intersection curve(s).
            This is typically only needed for the CFD surface mesh.

        """

        # if compList is not provided, we use all components
        if compNames is None:
            compNames = self.compNames

        # make the input at least 2d in case a single test point is passed
        points = np.atleast_2d(points)

        # before we do anything, we need to create surface ADTs
        # for which the user provided triangulated meshes
        for comp in compNames:
            # check if we have a trimesh for this component
            if self.comps[comp].triMesh:
                # Now we build the ADT using pySurf
                # Set bounding box for new tree
                BBox = np.zeros((2, 3))
                useBBox = False

                # dummy connectivity data for quad elements since we have all tris
                quadConn = np.zeros((0, 4))

                # Compute set of nodal normals by taking the average normal of all
                # elements surrounding the node. This allows the meshing algorithms,
                # for instance, to march in an average direction near kinks.
                nodal_normals = self.adtAPI.adtcomputenodalnormals(
                    self.comps[comp].nodes.T, self.comps[comp].triConnStack.T, quadConn.T
                )
                self.comps[comp].nodal_normals = nodal_normals.T

                # Create new tree (the tree itself is stored in Fortran level)
                self.adtAPI.adtbuildsurfaceadt(
                    self.comps[comp].nodes.T,
                    self.comps[comp].triConnStack.T,
                    quadConn.T,
                    BBox.T,
                    useBBox,
                    MPI.COMM_SELF.py2f(),
                    comp,
                )

        # create the pointset class
        self.points[ptName] = PointSet(points, comm=comm)

        for comp in self.compNames:
            # initialize the list for this component
            self.points[ptName].compMap[comp] = []
            self.points[ptName].compMapFlat[comp] = []

        # we now need to create the component mapping information
        for i in range(self.points[ptName].nPts):
            # initial flags
            inFFD = False
            proj = False
            projList = []

            # loop over components and check if this point is in a single BBox
            for comp in compNames:
                # apply a small tolerance for the bounding box in case points are coincident with the FFD
                boundTol = 1e-16
                xMin = self.comps[comp].xMin
                xMax = self.comps[comp].xMax
                xMin -= np.abs(xMin * boundTol) + boundTol
                xMax += np.abs(xMax * boundTol) + boundTol

                # check if inside
                if (
                    xMin[0] < points[i, 0] < xMax[0]
                    and xMin[1] < points[i, 1] < xMax[1]
                    and xMin[2] < points[i, 2] < xMax[2]
                ):
                    # add this component to the projection list
                    projList.append(comp)

                    # this point was not inside any other FFD before
                    if not inFFD:
                        inFFD = True
                        inComp = comp
                    # this point was inside another FFD, so we need to project it...
                    else:
                        # set the projection flag
                        proj = True

            # project this point to components, we need to set inComp string
            if proj:
                # set a high initial distance
                dMin2 = 1e10

                # loop over the components
                for comp in compNames:
                    # check if this component is in the projList
                    if comp in projList:
                        # check if we have an ADT:
                        if self.comps[comp].triMesh:
                            # Initialize reference values (see explanation above)
                            numPts = 1
                            dist2 = np.ones(numPts, dtype=self.dtype) * 1e10
                            xyzProj = np.zeros((numPts, 3), dtype=self.dtype)
                            normProjNotNorm = np.zeros((numPts, 3), dtype=self.dtype)

                            # Call projection function
                            _, _, _, _ = self.adtAPI.adtmindistancesearch(
                                points[i].T, comp, dist2, xyzProj.T, self.comps[comp].nodal_normals.T, normProjNotNorm.T
                            )

                            # if this is closer than the previous min, take this comp
                            if dist2 < dMin2:
                                dMin2 = dist2[0]
                                inComp = comp

                        else:
                            raise Error(
                                f"The point at (x, y, z) = ({points[i, 0]:.3f}, {points[i, 1]:.3f} {points[i, 2]:.3f})"
                                + f"in point set {ptName} is inside multiple FFDs but a triangulated mesh "
                                + f"for component {comp} is not provided to determine which component owns this point."
                            )

            # this point was inside at least one FFD. If it was inside multiple,
            # we projected it before to figure out which component it should belong to
            if inFFD:
                # we can add the point index to the list of points inComp owns
                self.points[ptName].compMap[inComp].append(i)

                # also create a flattened version of the compMap
                for j in range(3):
                    self.points[ptName].compMapFlat[inComp].append(3 * i + j)

            # this point is outside any FFD...
            else:
                raise Error(
                    f"The point at (x, y, z) = ({points[i, 0]:.3f}, {points[i, 1]:.3f} {points[i, 2]:.3f}) "
                    + f"in point set {ptName} is not inside any FFDs."
                )

        # using the mapping array, add the pointsets to respective DVGeo objects
        for comp in self.compNames:
            compMap = self.points[ptName].compMap[comp]
            self.comps[comp].DVGeo.addPointSet(points[compMap], ptName, **kwargs)

        # check if this pointset will get the IC treatment
        if applyIC:
            # loop over the intersections and add pointsets
            for IC in self.intersectComps:
                IC.addPointSet(points, ptName, self.points[ptName].compMap, comm)

        # finally, we can deallocate the ADTs
        for comp in compNames:
            if self.comps[comp].triMesh:
                self.adtAPI.adtdeallocateadts(comp)

        # mark this pointset as up to date
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

        # loop over the components and set the values
        for comp in self.compNames:
            self.comps[comp].DVGeo.setDesignVars(dvDict)

        # set the current config as None, we cannot know what config the user wants until the update call
        self.curConfig = None
        self.updateICs = True

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

        dvDict = {}
        # we need to loop over each DVGeo object and get the DVs
        for comp in self.compNames:
            dvDictComp = self.comps[comp].DVGeo.getValues()
            # we need to loop over these DVs
            for k, v in dvDictComp.items():
                dvDict[k] = v

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

        # check if the ICs are up to date with the current config
        if self.updateICs or config != self.curConfig:
            self._setICSurfaces(config)
            self.updateICs = False
            self.curConfig = config

        # get the new points
        newPts = np.zeros((self.points[ptSetName].nPts, 3), dtype=self.dtype)

        # we first need to update all points with their respective DVGeo objects
        for comp in self.compNames:
            ptsComp = self.comps[comp].DVGeo.update(ptSetName, config=config)

            # now save this info with the pointset mapping
            ptMap = self.points[ptSetName].compMap[comp]
            newPts[ptMap] = ptsComp

        # get the delta
        delta = newPts - self.points[ptSetName].points

        # then apply the intersection treatment
        for IC in self.intersectComps:
            # check if this IC is active for this ptSet
            if ptSetName in IC.points:
                delta = IC.update(ptSetName, delta)

        # now we are ready to take the delta which may be modified by the intersections
        newPts = self.points[ptSetName].points + delta

        # now, project the points that were warped back onto the trimesh
        for IC in self.intersectComps:
            if IC.projectFlag and ptSetName in IC.points:
                # new points will be modified in place using the newPts array
                IC.project(ptSetName, newPts)

        # set the pointset up to date
        self.updated[ptSetName] = True

        return newPts

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
        nDV = 0
        for comp in self.compNames:
            nDV += self.comps[comp].DVGeo.getNDV()
        return nDV

    def getVarNames(self, pyOptSparse=False):
        """
        Return a list of the design variable names.
        This is typically used when specifying a ``wrt=`` argument for pyOptSparse.

        Examples
        --------
        >>> optProb.addCon(.....wrt=DVGeo.getVarNames())

        """
        dvNames = []
        # create a list of DVs from each comp
        for comp in self.compNames:
            # first get the list of DVs from this component
            varNames = self.comps[comp].DVGeo.getVarNames()

            # add the component DVs to the full list
            dvNames.extend(varNames)

        return dvNames

    def totalSensitivity(self, dIdpt, ptSetName, comm=None, config=None):
        """
        This function computes sensitivity information.

        Specificly, it computes the following:
        :math:`\\frac{dX_{pt}}{dX_{DV}}^T \\frac{dI}{d_{pt}}`

        Parameters
        ----------
        dIdpt : array of size (Npt, 3) or (N, Npt, 3)

            This is the total derivative of the objective or function
            of interest with respect to the coordinates in
            'ptSetName'. This can be a single array of size (Npt, 3)
            **or** a group of N vectors of size (Npt, 3, N). If you
            have many to do, it is faster to do many at once.

        ptSetName : str
            The name of set of points we are dealing with

        comm : MPI.IntraComm, optional
            The communicator to use to reduce the final derivative.
            If comm is None, no reduction takes place.

        config : str or list, optional
            Define what configurations this design variable will be applied to
            Use a string for a single configuration or a list for multiple
            configurations. The default value of None implies that the design
            variable appies to *ALL* configurations.


        Returns
        -------
        dIdxDict : dict
            The dictionary containing the derivatives, suitable for pyOptSparse.

        Notes
        -----
        The ``child`` and ``nDVStore`` options are only used
        internally and should not be changed by the user.

        """

        # TODO temporary solution. Ideally, we may want to track configs per pointset
        # need to run a full update here
        # this will set self.updateICs to False and
        # self.curConfig to config. It will also
        # update the underlying data for the intersections
        self.update(ptSetName, config=config)

        # Compute the total Jacobian for this point set
        self._computeTotalJacobian(ptSetName, config)

        # Make dIdpt at least 3D
        if len(dIdpt.shape) == 2:
            dIdpt = np.array([dIdpt])
        N = dIdpt.shape[0]

        # create a dictionary to save total sensitivity info that might come out of the ICs
        compSensList = []

        # if we projected points for any intersection treatment,
        # we need to propagate the derivative seed of the projected points
        # back to the seeds for the initial points we get after ID-warping
        for IC in self.intersectComps:
            if IC.projectFlag and ptSetName in IC.points:
                # initialize the seed contribution to the intersection seam and feature curves from project_b
                IC.seamBarProj[ptSetName] = np.zeros((N, IC.seam0.shape[0], IC.seam0.shape[1]))

                # we pass in dIdpt and the intersection object, along with pointset information
                # the intersection object adjusts the entries corresponding to projected points
                # and passes back dIdpt in place.
                compSens = IC.project_b(ptSetName, dIdpt, comm, config)

                # append this to the dictionary list...
                compSensList.append(compSens)

        # do the transpose multiplication

        if self.debug:
            print(f"[{self.comm.rank}] finished project_b")

        # we need to go through all ICs bec even though some procs might not have points on the intersection,
        # communication is easier and we can reduce compSens as we compute them
        for IC in self.intersectComps:
            if ptSetName in IC.points:
                compSens = IC.sens(dIdpt, ptSetName, comm, config)
                # save the sensitivities from the intersection stuff
                compSensList.append(compSens)

        if self.debug:
            print(f"[{self.comm.rank}] finished IC.sens")

        # reshape the dIdpt array from [N] * [nPt] * [3] to  [N] * [nPt*3]
        dIdpt = dIdpt.reshape((dIdpt.shape[0], dIdpt.shape[1] * 3))

        # jacobian for the pointset
        jac = self.points[ptSetName].jac

        # this is the mat-vec product for the remaining seeds.
        # this only contains the effects of the FFD motion,
        # projections and intersections are handled separately in compSens
        dIdxT_local = jac.T.dot(dIdpt.T)
        dIdx_local = dIdxT_local.T

        # If we have a comm, globaly reduce with sum
        if comm:
            dIdx = comm.allreduce(dIdx_local, op=MPI.SUM)
        else:
            dIdx = dIdx_local

        # use respective DVGeo's convert to dict functionality
        dIdxDict = OrderedDict()
        dvOffset = 0
        for comp in self.compNames:
            DVGeo = self.comps[comp].DVGeo
            nDVComp = DVGeo.getNDV()

            # we only do this if this component has at least one DV
            if nDVComp > 0:
                # this part of the sensitivity matrix is owned by this dvgeo
                dIdxComp = DVGeo.convertSensitivityToDict(dIdx[:, dvOffset : dvOffset + nDVComp])

                for k, v in dIdxComp.items():
                    dIdxDict[k] = v

                # also increment the offset
                dvOffset += nDVComp

        # finally, we can add the contributions from triangulated component meshes
        for compSens in compSensList:
            # loop over the items of compSens, which are guaranteed to be in dIdxDict
            for k, v in compSens.items():
                # these will bring in effects from projections and intersection computations
                dIdxDict[k] += v

        if self.debug:
            print(f"[{self.comm.rank}] finished DVGeo.totalSensitivity")

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

        # If no list was provided, we use all components
        if comps is None:
            comps = self.compNames

        # We can simply loop over all DV objects and call their respective addVariablesPyOpt function
        for comp in comps:
            self.comps[comp].DVGeo.addVariablesPyOpt(
                optProb,
                globalVars=globalVars,
                localVars=localVars,
                sectionlocalVars=sectionlocalVars,
                ignoreVars=ignoreVars,
                freezeVars=freezeVars,
            )

    def getLocalIndex(self, iVol, comp):
        """Return the local index mapping that points to the global coefficient list for a given volume.

        Parameters
        ----------

        iVol : int
            Index specifying the FFD volume.

        comp : str
            Name of the component.

        """

        # Call this on the component DVGeo
        DVGeo = self.comps[comp].DVGeo
        return DVGeo.FFD.topo.lIndex[iVol].copy()

    # ----------------------------------------------------------------------
    #        THE REMAINDER OF THE FUNCTIONS NEED NOT BE CALLED BY THE USER
    # ----------------------------------------------------------------------

    def _readCGNSFile(self, filename):
        # this function reads the unstructured CGNS grid in filename and returns
        # node coordinates and element connectivities.
        # Here, only the root proc reads the cgns file, broadcasts node and connectivity info.

        # only root proc reads the file
        if self.comm.rank == 0:
            print(f"Reading file {filename}")
            # use the default routine in tsurftools
            nodes, sectionDict = tsurf_tools.getCGNSsections(filename, comm=MPI.COMM_SELF)
            print("Finished reading the cgns file")

            # Convert the nodes to complex if necessary
            nodes = nodes.astype(self.dtype)

            triConn = {}
            triConnStack = np.zeros((0, 3), dtype=np.int8)
            barsConn = {}

            for part in sectionDict:
                if "triaConnF" in sectionDict[part].keys():
                    # this is a surface, read the tri connectivities
                    triConn[part.lower()] = sectionDict[part]["triaConnF"]
                    triConnStack = np.vstack((triConnStack, sectionDict[part]["triaConnF"]))

                if "barsConn" in sectionDict[part].keys():
                    # this is a curve, save the curve connectivity
                    barsConn[part.lower()] = sectionDict[part]["barsConn"]

            print(f"The {filename} mesh has {len(nodes)} nodes and {len(triConnStack)} elements.")
        else:
            # create these to recieve the data
            nodes = None
            triConn = None
            triConnStack = None
            barsConn = None

        # each proc gets the nodes and connectivities
        nodes = self.comm.bcast(nodes, root=0)
        triConn = self.comm.bcast(triConn, root=0)
        triConnStack = self.comm.bcast(triConnStack, root=0)
        barsConn = self.comm.bcast(barsConn, root=0)

        return nodes, triConn, triConnStack, barsConn

    def _readP3DCurve(self, filename):

        def readplot3d(fileName):
            order="f"
            binary = False
            f = open(fileName)
            nVol = readNValues(f, 1, "int", False)[0]
            sizes = readNValues(f, nVol * 3, "int", False).reshape((nVol, 3))
            blocks = []
            for i in range(nVol):
                cur_size = sizes[i, 0] * sizes[i, 1] * sizes[i, 2]
                blocks.append(np.zeros([sizes[i, 0], sizes[i, 1], sizes[i, 2], 3]))
                for idim in range(3):
                    blocks[-1][:, :, :, idim] = readNValues(f, cur_size, "float", binary).reshape(
                        (sizes[i, 0], sizes[i, 1], sizes[i, 2]), order=order
                    )
            f.close()

            return blocks

        if self.comm.rank == 0:
            p3dblocks = readplot3d(filename)

            # each "block" has implied connectivity and ordered set of nodes
            # we will stack these nodes and connectivities in this array
            nodes = np.zeros((0, 3))
            barsConn = np.zeros((0, 2), dtype="int32")
            for block in p3dblocks:
                # squeeze out the unused axes. this is just a curve so its nPoint, 3
                blockNodes = block.squeeze()
                nNode = blockNodes.shape[0]
                # then create a local connectivity array for this curve only
                barsConnLocal = np.zeros((nNode - 1, 2), dtype="int32")
                barsConnLocal[:, 0] = np.arange(nNode - 1)
                barsConnLocal[:, 1] = barsConnLocal[:, 0] + 1

                # now increment by the last node count
                barsConnLocal += nodes.shape[0]

                # first vstack the nodes
                nodes = np.vstack((nodes, blockNodes))

                # finally, vstack with barsConn
                barsConn = np.vstack((barsConn, barsConnLocal))

        else:
            # create these to recieve the data
            nodes = None
            barsConn = None

        # each proc gets the nodes and connectivities
        nodes = self.comm.bcast(nodes, root=0)
        barsConn = self.comm.bcast(barsConn, root=0)

        return nodes, barsConn

    def _computeTotalJacobian(self, ptSetName, config):
        """
        This routine computes the total jacobian. It takes the jacobians
        from respective DVGeo objects and also computes the jacobians for
        the intersection seams. We then use this information in the
        totalSensitivity function.

        """

        # number of design variables
        nDV = self.getNDV()

        # Initialize the Jacobian as a LIL matrix because this is convenient for indexing
        jac = sparse.lil_matrix((self.points[ptSetName].nPts * 3, nDV))

        # ptset
        ptSet = self.points[ptSetName]

        dvOffset = 0
        # we need to call computeTotalJacobian from all comps and get the jacobians for this pointset
        for comp in self.compNames:
            # number of design variables
            nDVComp = self.comps[comp].DVGeo.getNDV()

            # call the function to compute the total jacobian
            self.comps[comp].DVGeo.computeTotalJacobian(ptSetName, config)

            if self.comps[comp].DVGeo.JT[ptSetName] is not None:
                # Get the component Jacobian
                compJ = self.comps[comp].DVGeo.JT[ptSetName].T

                # Set the block of the full Jacobian associated with this component
                jac[ptSet.compMapFlat[comp], dvOffset : dvOffset + nDVComp] = compJ

            # increment the offset
            dvOffset += nDVComp

        # Convert to CSR format because this is better for arithmetic
        jac = sparse.csr_matrix(jac)

        # now we can save this jacobian in the pointset
        ptSet.jac = jac

    def _setICSurfaces(self, config):
        # updates the ICs with the given config

        # We need to give the updated coordinates to each of the
        # intersectComps (if we have any) so they can update the new intersection curve
        for IC in self.intersectComps:
            IC.setSurface(self.comm, config)


class component:
    def __init__(self, name, DVGeo, nodes, triConn, triConnStack, barsConn, xMin, xMax, triMeshData, pointSetKwargs):
        # save the info
        self.name = name
        self.DVGeo = DVGeo
        self.nodes = nodes
        self.triConn = triConn
        self.triConnStack = triConnStack
        self.barsConn = barsConn
        self.xMin = xMin
        self.xMax = xMax
        self.triMeshData = triMeshData
        self.pointSetKwargs = pointSetKwargs

        # also a dictionary for DV names
        self.dvDict = {}

        # set a flag for triangulated meshes
        if nodes is None:
            self.triMesh = False
        else:
            self.triMesh = True

        self.triMeshAdded = False
        self.triMeshName = f"triMesh_{name}"

    def addTriMeshPoints(self, comm):
        disp = self.triMeshData["disp"]
        procNodes = self.nodes[disp[comm.rank] : disp[comm.rank + 1]]
        self.DVGeo.addPointSet(procNodes, self.triMeshName, **self.pointSetKwargs)

    def updateTriMesh(self, comm, config):
        if not self.triMeshAdded:
            self.addTriMeshPoints(comm)
            self.triMeshAdded = True

        # We need the full triangulated surface for this component
        # Get the stored processor splitting information
        sizes = self.triMeshData["sizes"]
        disp = self.triMeshData["disp"]
        nPts = disp[-1]

        # Update the triangulated surface mesh to get the points on this processor
        procNodes = self.DVGeo.update(self.triMeshName, config=config)

        # Create the send buffer
        procNodes = procNodes.flatten()
        sendbuf = [procNodes, sizes[comm.rank] * 3]

        # Set the appropriate type for the receiving buffer
        if procNodes.dtype == float:
            mpiType = MPI.DOUBLE
        elif procNodes.dtype == complex:
            mpiType = MPI.DOUBLE_COMPLEX

        # Create the receiving buffer
        globalNodes = np.zeros(nPts * 3, dtype=procNodes.dtype)
        recvbuf = [globalNodes, sizes * 3, disp[0:-1] * 3, mpiType]

        # Allgather the updated coordinates
        comm.Allgatherv(sendbuf, recvbuf)

        # Reshape into a nPts, 3 array
        self.nodes = globalNodes.reshape((nPts, 3))


class PointSet:
    def __init__(self, points, comm):
        self.points = points
        self.nPts = len(self.points)
        self.compMap = OrderedDict()
        self.compMapFlat = OrderedDict()
        self.comm = comm
