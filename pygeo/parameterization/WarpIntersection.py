# Standard Python modules
from collections import OrderedDict

# External modules
from baseclasses.utils import Error
from mpi4py import MPI
import numpy as np
from pathlib import Path

try:
    # External modules
    from pysurf import (
        adtAPI,
        adtAPI_cs,
        curveSearchAPI,
        curveSearchAPI_cs,
        intersectionAPI,
        intersectionAPI_cs,
        tecplot_interface,
        utilitiesAPI,
        utilitiesAPI_cs,
    )

    pysurfInstalled = True
except ImportError:
    pysurfInstalled = False


class WarpedIntersection:
    def __init__(
        self,
        compA,
        compB,
        dStarA,
        dStarB,
        nodes,
        barsConn,
        DVGeo,
        anisotropy,
        blendOrder,
        debug,
        dtype,
        idx,
    ):
        """
        Class to store information required for an intersection.
        Here, we use some Fortran code from pySurf.
        Internally, we store the indices and weights of the points that this intersection will modify.
        This code is not super efficient because it is in Python.

        See the documentation for ``addIntersection`` in DVGeometryMulti for the API.

        """

        # Flag for debug ouput
        self.debug = debug
        self.idx = idx

        # same communicator with DVGeo
        self.comm = DVGeo.comm
        self.curConfigText = ""

        self.name = f"{compA}_{compB}_{self.idx}_int_{self.idx}"
        self.debug_dir = f"./dvgeomulti_debug_outputs/{self.name}"
        if self.comm.rank == 0 and self.debug:
            Path(self.debug_dir).mkdir(parents=True, exist_ok=True)
        self.comm.Barrier()

        # define epsilon as a small value to prevent division by zero in the inverse distance computation
        self.eps = 1e-20

        # counter for outputting curves etc at each update
        self.counter = 0

        # Set real or complex Fortran APIs
        self.dtype = dtype
        if dtype == float:
            self.adtAPI = adtAPI.adtapi
            self.curveSearchAPI = curveSearchAPI.curvesearchapi
            self.intersectionAPI = intersectionAPI.intersectionapi
            self.utilitiesAPI = utilitiesAPI.utilitiesapi
            self.mpiType = MPI.DOUBLE
        elif dtype == complex:
            self.adtAPI = adtAPI_cs.adtapi
            self.curveSearchAPI = curveSearchAPI_cs.curvesearchapi
            self.intersectionAPI = intersectionAPI_cs.intersectionapi
            self.utilitiesAPI = utilitiesAPI_cs.utilitiesapi
            self.mpiType = MPI.DOUBLE_COMPLEX

        # names of compA and compB must be provided
        self.compA = DVGeo.comps[compA]
        self.compB = DVGeo.comps[compB]

        self.dStarA = dStarA
        self.dStarB = dStarB
        self.points = OrderedDict()

        self.blendOrder = blendOrder

        # we never do projections in this intersection class
        self.projectFlag = False

        # Save anisotropy list
        self.anisotropy = anisotropy

        # save the initial intersection curve
        self.seam0 = nodes.copy()
        # make 2 more copies of the seam. we will use these to calculate separate deltas for each comp
        self.seamA = nodes.copy()
        self.seamB = nodes.copy()
        self.seamConn = barsConn.copy()

        self.pointsAdded = False

    def setSurface(self, comm, config):
        """This set the new udpated surface on which we need to compute the new intersection curve"""

        if config is not None:
            self.curConfigText = f"_{config}"

        # add the intersection curve to each component dvgeos separately
        if not self.pointsAdded:
            self.compAIntName = f"{self.name}_{self.idx}_int_coords_A"
            self.compBIntName = f"{self.name}_{self.idx}_int_coords_B"
            self.compA.DVGeo.addPointSet(self.seam0.copy(), self.compAIntName)
            self.compB.DVGeo.addPointSet(self.seam0.copy(), self.compBIntName)
            self.pointsAdded = True

        # get the updated surface coordinates
        self._updateSeamCoords(config)


    def addPointSet(self, pts, ptSetName, compMap, comm):
        # TODO
        # Figure out which points this intersection object has to deal with

        # Use pySurf to project the point on curve
        # Get number of points
        nPoints = len(pts)

        # Initialize references if user provided none
        dist2 = np.ones(nPoints, dtype=self.dtype) * 1e10
        xyzProj = np.zeros((nPoints, 3), dtype=self.dtype)
        tanProj = np.zeros((nPoints, 3), dtype=self.dtype)
        elemIDs = np.zeros((nPoints), dtype="int32")

        # Only call the Fortran code if we have at least one point
        if nPoints > 0:
            # This will modify xyzProj, tanProj, dist2, and elemIDs if we find better projections than dist2.
            # Remember that we should adjust some indices before calling the Fortran code
            # Remember to use [:] to don't lose the pointer (elemIDs is an input/output variable)
            elemIDs[:] = (
                elemIDs + 1
            )  # (we need to do this separetely because Fortran will actively change elemIDs contents.
            self.curveSearchAPI.mindistancecurve(
                pts.T, self.seam0.T, self.seamConn.T + 1, xyzProj.T, tanProj.T, dist2, elemIDs
            )

            # Adjust indices back to Python standards
            elemIDs[:] = elemIDs - 1

        # dist2 has the array of squared distances
        d = np.sqrt(dist2)

        indices = []
        factors = []
        for i in range(len(pts)):
            # figure out which component this point is mapped to
            if i in compMap[self.compA.name]:
                # component A owns this
                dStar = self.dStarA
            else:
                # comp B owns this point
                dStar = self.dStarB

            if dStar is not None and d[i] < dStar:
                x = d[i] / dStar
                x3 = x * x * x
                x4 = x3 * x
                x5 = x4 * x
                factor = 10 * x3 - 15 * x4 + 6 * x5

                # # TODO update this to add a const buffer ?
                # if d[i] < halfdStar:
                #     factor = 0.5 * (d[i] / halfdStar) ** self.blendOrder
                # else:
                #     factor = 0.5 * (2 - ((dStar - d[i]) / halfdStar) ** self.blendOrder)

                # TODO save the nearest n elements?

                # Save the index and factor
                indices.append(i)
                factors.append(factor)

        # Get all points included in the intersection computation
        intersectPts = pts[indices]
        nPoints = len(intersectPts)

        # Save the affected indices and the factor in the little dictionary
        self.points[ptSetName] = {
            "pts": pts.copy(),
            "indices": indices,
            "factors": factors,
            "compMap": compMap,
        }


    def update(self, ptSetName, delta):
        """Update the delta in ptSetName with our correction. The delta need
        to be supplied as we will be changing it and returning them
        """

        # first figure out if any of the dStars is set to None. this will trigger a modified algorithm
        if self.dStarA is None and self.dStarB is None:
            # this should not happen, we can only set one to None. error out
            raise Error("both dStarA and dStarB cannot be set to None. Either set one of them to None or neither to None.")
        elif self.dStarA is None or self.dStarB is None:
            # this means we are using the modified algorithm, modify the delta multipliers
            # this works because the component with a None dStar does not get any warped points
            deltaFull = 0.0
            deltaWarp = 1.0
        else:
            # this is the baseline algorithm. the math we ultimately want is:
            # delta[j] += (1 - factors[i]) * interp
            # to achieve this, modify the factors
            deltaFull = 1.0
            deltaWarp = 0.0

        # original coordinates of the added pointset
        pts = self.points[ptSetName]["pts"]
        # indices of the points that get affected by this intersection
        indices = self.points[ptSetName]["indices"]
        # factors for each node in pointSet
        factors = self.points[ptSetName]["factors"]
        # compMap for these points in the original array
        compMap = self.points[ptSetName]["compMap"]

        # coordinates for the remeshed curves
        # we use the initial seam coordinates here
        coor = self.seam0
        conn = self.seamConn

        # deltas for each point (nNode, 3) in size
        drA = self.seamA - self.seam0
        drB = self.seamB - self.seam0

        # Get the two end points for the line elements
        # TODO potentially update this so that every point only considers a subset of the conn array which is the nearest N elements?
        r0 = coor[conn[:, 0]]
        r1 = coor[conn[:, 1]]

        # Get the deltas for two end points
        dr0A = drA[conn[:, 0]]
        dr1A = drA[conn[:, 1]]
        dr0B = drB[conn[:, 0]]
        dr1B = drB[conn[:, 1]]

        # Compute the lengths of each element in each coordinate direction
        length_x = r1[:, 0] - r0[:, 0]
        length_y = r1[:, 1] - r0[:, 1]
        length_z = r1[:, 2] - r0[:, 2]

        # Compute the 'a' coefficient
        a = (length_x) ** 2 + (length_y) ** 2 + (length_z) ** 2

        # Compute the total length of each element
        length = np.sqrt(a)

        # loop over the points that get affected
        for i in range(len(factors)):
            # j is the index oxxxf the point in the full set we are working with.
            j = indices[i]

            # figure out which components delta we will use
            # we flip this around here.
            # for points on compA, we use the compB deltas, and vice versa
            if j in compMap[self.compA.name]:
                dr0 = dr0B
                dr1 = dr1B
            else:
                dr0 = dr0A
                dr1 = dr1A

            # coordinates of the original point
            rp = pts[j]

            # Run vectorized weighted interpolation

            # Compute the distances from the point being updated to the first end point of each element
            # The distances are scaled by the user-specified anisotropy in each direction
            dist_x = (r0[:, 0] - rp[0]) * self.anisotropy[0]
            dist_y = (r0[:, 1] - rp[1]) * self.anisotropy[1]
            dist_z = (r0[:, 2] - rp[2]) * self.anisotropy[2]

            # Compute b and c coefficients
            b = 2 * (length_x * dist_x + length_y * dist_y + length_z * dist_z)
            c = dist_x**2 + dist_y**2 + dist_z**2

            # Compute some recurring terms

            # The discriminant can be zero or negative, but it CANNOT be positive
            # This is because the quadratic that defines the distance from the line cannot have two roots
            # If the point is on the line, the quadratic will have a single root
            disc = b * b - 4 * a * c

            # Clip a + b + c might because it might be negative 1e-20 or so
            # Analytically, it cannot be negative
            sabc = np.sqrt(np.maximum(a + b + c, 0.0))
            sc = np.sqrt(c)

            # Compute denominators for the integral evaluations
            # We clip these values so that they are at max -eps to prevent them from getting a value of zero.
            # disc <= 0, sabc and sc >= 0, therefore the den1 and den2 should be <=0.
            # The clipping forces these terms to be <= -eps
            den1 = np.minimum(disc * sabc, -self.eps)
            den2 = np.minimum(disc * sc, -self.eps)

            # integral evaluations
            eval1 = (-2 * (2 * a + b) / den1 + 2 * b / den2) * length
            eval2 = ((2 * b + 4 * c) / den1 - 4 * c / den2) * length

            # denominator only gets one integral
            den = np.sum(eval1)

            # do each direction separately
            interp = np.zeros(3, dtype=self.dtype)
            for iDim in range(3):
                # numerator gets two integrals with the delta components
                num = np.sum((dr1[:, iDim] - dr0[:, iDim]) * eval2 + dr0[:, iDim] * eval1)
                # final result
                interp[iDim] = num / den

            # Now the delta is replaced by 1-factor times the weighted
            # interp of the seam * factor of the original:
            delta[j] = delta[j] * deltaFull + delta[j] * factors[i] * deltaWarp + (1 - factors[i]) * interp

        return delta

    def sens(self, dIdPt, ptSetName, comm, config):
        # Return the reverse accumulation of dIdpt on the seam
        # nodes. Also modifies the dIdp array accordingly.

        # first figure out if any of the dStars is set to None. this will trigger a modified algorithm
        if self.dStarA is None and self.dStarB is None:
            # this should not happen, we can only set one to None. error out
            raise Error("both dStarA and dStarB cannot be set to None. Either set one of them to None or neither to None.")
        elif self.dStarA is None or self.dStarB is None:
            # this means we are using the modified algorithm, modify the delta multipliers
            # this works because the component with a None dStar does not get any warped points
            deltaFull = 0.0
            deltaWarp = 1.0
        else:
            # this is the baseline algorithm. the math we ultimately want is:
            # delta[j] += (1 - factors[i]) * interp
            # to achieve this, modify the factors
            deltaFull = 1.0
            deltaWarp = 0.0

        # original coordinates of the added pointset
        pts = self.points[ptSetName]["pts"]
        # indices of the points that get affected by this intersection
        indices = self.points[ptSetName]["indices"]
        # factors for each node in pointSet
        factors = self.points[ptSetName]["factors"]
        # compMap for these points in the original array
        compMap = self.points[ptSetName]["compMap"]

        # coordinates for the remeshed curves
        # we use the initial seam coordinates here
        coor = self.seam0
        conn = self.seamConn

        # Get the two end points for the line elements
        r0 = coor[conn[:, 0]]
        r1 = coor[conn[:, 1]]

        # Compute the lengths of each element in each coordinate direction
        length_x = r1[:, 0] - r0[:, 0]
        length_y = r1[:, 1] - r0[:, 1]
        length_z = r1[:, 2] - r0[:, 2]

        # Compute the 'a' coefficient
        a = (length_x) ** 2 + (length_y) ** 2 + (length_z) ** 2

        # Compute the total length of each element
        length = np.sqrt(a)

        # if we are handling more than one function,
        # seamBar will contain the seeds for each function separately
        seamABar = np.zeros((dIdPt.shape[0], self.seam0.shape[0], self.seam0.shape[1]))
        seamBBar = np.zeros((dIdPt.shape[0], self.seam0.shape[0], self.seam0.shape[1]))

        for i in range(len(factors)):
            # j is the index of the point in the full set we are working with.
            j = indices[i]

            # figure out which components delta we will use
            # we flip this around here.
            # for points on compA, we use the compB deltas, and vice versa
            if j in compMap[self.compA.name]:
                seamBar = seamBBar
            else:
                seamBar = seamABar

            # coordinates of the original point
            rp = pts[j]

            # Compute the distances from the point being updated to the first end point of each element
            # The distances are scaled by the user-specified anisotropy in each direction
            dist_x = (r0[:, 0] - rp[0]) * self.anisotropy[0]
            dist_y = (r0[:, 1] - rp[1]) * self.anisotropy[1]
            dist_z = (r0[:, 2] - rp[2]) * self.anisotropy[2]

            # Compute b and c coefficients
            b = 2 * (length_x * dist_x + length_y * dist_y + length_z * dist_z)
            c = dist_x**2 + dist_y**2 + dist_z**2

            # Compute some reccurring terms
            disc = b * b - 4 * a * c
            sabc = np.sqrt(np.maximum(a + b + c, 0.0))
            sc = np.sqrt(c)

            # Compute denominators for the integral evaluations
            den1 = np.minimum(disc * sabc, -self.eps)
            den2 = np.minimum(disc * sc, -self.eps)

            # integral evaluations
            eval1 = (-2 * (2 * a + b) / den1 + 2 * b / den2) * length
            eval2 = ((2 * b + 4 * c) / den1 - 4 * c / den2) * length

            # denominator only gets one integral
            den = np.sum(eval1)

            evalDiff = eval1 - eval2

            for k in range(dIdPt.shape[0]):
                # This is the local seed (well the 3 seeds for the point)
                localVal = dIdPt[k, j, :] * (1 - factors[i])

                # do each direction separately
                for iDim in range(3):
                    # seeds for the r0 point
                    seamBar[k, conn[:, 0], iDim] += localVal[iDim] * evalDiff / den

                    # seeds for the r1 point
                    seamBar[k, conn[:, 1], iDim] += localVal[iDim] * eval2 / den

                # we need to modify dIdPt as well, based on what warping mode we are in
                dIdPt[k, j, :] = dIdPt[k, j, :] * deltaFull + dIdPt[k, j, :] * deltaWarp * factors[i]

        # seamBar is the bwd seeds for the intersection curve...
        # it is N,nseampt,3 in size
        # now call the reverse differentiated seam computation
        compSens = self._getIntersectionSeam_b(seamABar, seamBBar, comm, config)

        return compSens


    def _updateSeamCoords(self, config):
        # this code returns the updated coordinates

        # first comp a
        self.seamA = self.compA.DVGeo.update(self.compAIntName, config=config)

        # then comp b
        self.seamB = self.compB.DVGeo.update(self.compBIntName, config=config)

        # Output the intersection curve
        if self.comm.rank == 0 and self.debug:
            curvename = f"{self.debug_dir}/{self.compA.name}_{self.compB.name}_{self.counter}{self.curConfigText}"
            tecplot_interface.writeTecplotFEdata(self.seamB, self.seamConn, curvename, curvename)
            self.counter += 1

        return

    def _getIntersectionSeam_b(self, seamABar, seamBBar, comm, config):
        # we get the seeds on the two copies of the seam. run these through the respective DVGeos and return the final dictionary

        # get the total sensitivities from both components
        compSens_local = {}
        compSensA = self.compA.DVGeo.totalSensitivity(seamABar, self.compAIntName, config=config)
        for k, v in compSensA.items():
            compSens_local[k] = v

        compSensB = self.compB.DVGeo.totalSensitivity(seamBBar, self.compBIntName, config=config)
        for k, v in compSensB.items():
            compSens_local[k] = v

        # finally sum the results across procs if we are provided with a comm
        if comm:
            compSens = {}
            # because the results are in a dictionary, we need to loop over the items and sum
            for k, v in compSens_local.items():
                compSens[k] = comm.allreduce(v, op=MPI.SUM)
        else:
            # we can just pass the dictionary
            compSens = compSens_local

        return compSens