# External modules
import numpy as np

# Local modules
from .. import geo_utils
from .baseConstraint import GeometricConstraint


class ThicknessConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of thickness
    constraints. One of these objects is created each time a
    addThicknessConstraints2D or addThicknessConstraints1D call is
    made. The user should not have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 2, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.scaled = scaled

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.D0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            self.D0[i] = geo_utils.norm.euclideanNorm(self.coords[2 * i] - self.coords[2 * i + 1])

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        D = np.zeros(self.nCon)
        for i in range(self.nCon):
            D[i] = geo_utils.norm.euclideanNorm(self.coords[2 * i] - self.coords[2 * i + 1])
            if self.scaled:
                D[i] /= self.D0[i]
        funcs[self.name] = D

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dTdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))

            for i in range(self.nCon):
                p1b, p2b = geo_utils.eDist_b(self.coords[2 * i, :], self.coords[2 * i + 1, :])
                if self.scaled:
                    p1b /= self.D0[i]
                    p2b /= self.D0[i]
                dTdPt[i, 2 * i, :] = p1b
                dTdPt[i, 2 * i + 1, :] = p2b

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dTdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class ThicknessToChordConstraint(GeometricConstraint):
    """
    ThicknessToChordConstraint represents of a set of
    thickess-to-chord ratio constraints. One of these objects is
    created each time a addThicknessToChordConstraints2D or
    addThicknessToChordConstraints1D call is made. The user should not
    have to deal with this class directly.
    """

    def __init__(self, name, coords, lower, upper, scale, DVGeo, addToPyOpt, compNames):
        super().__init__(name, len(coords) // 4, lower, upper, scale, DVGeo, addToPyOpt)
        self.coords = coords

        # First thing we can do is embed the coordinates into DVGeo
        # with the name provided:
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Now get the reference lengths
        self.ToC0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            t = np.linalg.norm(self.coords[4 * i] - self.coords[4 * i + 1])
            c = np.linalg.norm(self.coords[4 * i + 2] - self.coords[4 * i + 3])
            self.ToC0[i] = t / c

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(self.name, config=config)
        ToC = np.zeros(self.nCon)
        for i in range(self.nCon):
            t = geo_utils.eDist(self.coords[4 * i], self.coords[4 * i + 1])
            c = geo_utils.eDist(self.coords[4 * i + 2], self.coords[4 * i + 3])
            ToC[i] = (t / c) / self.ToC0[i]

        funcs[self.name] = ToC

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dToCdPt = np.zeros((self.nCon, self.coords.shape[0], self.coords.shape[1]))

            for i in range(self.nCon):
                t = geo_utils.eDist(self.coords[4 * i], self.coords[4 * i + 1])
                c = geo_utils.eDist(self.coords[4 * i + 2], self.coords[4 * i + 3])

                p1b, p2b = geo_utils.eDist_b(self.coords[4 * i, :], self.coords[4 * i + 1, :])
                p3b, p4b = geo_utils.eDist_b(self.coords[4 * i + 2, :], self.coords[4 * i + 3, :])

                dToCdPt[i, 4 * i, :] = p1b / c / self.ToC0[i]
                dToCdPt[i, 4 * i + 1, :] = p2b / c / self.ToC0[i]
                dToCdPt[i, 4 * i + 2, :] = (-p3b * t / c**2) / self.ToC0[i]
                dToCdPt[i, 4 * i + 3, :] = (-p4b * t / c**2) / self.ToC0[i]

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dToCdPt, self.name, config=config)

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords), len(self.coords) // 2))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class KSMaxThicknessToChordConstraint(GeometricConstraint):
    """
    KSMaxThicknessToChordConstraint represents the maximum of a
    set of thickess-to-chord ratio constraints.  The chord is computed
    as the Euclidean distance from the provided leading edge to trailing
    edge points.  Each thickness is divided by the chord distance and
    then the max value is computed using a KS function.

    One of these objects is created each time a
    addKSMaxThicknessToChordConstraints call is made. The user should
    not have to deal with this class directly.
    """

    def __init__(self, name, coords, lePt, tePt, rho, divideByChord, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        self.nPoint = len(coords) // 2
        super().__init__(name, 1, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.leTePts = np.array([lePt, tePt])
        self.scaled = scaled
        self.rho = rho
        self.divideByChord = divideByChord

        # Embed the coordinates
        self.DVGeo.addPointSet(self.coords, f"{self.name}_coords", compNames=compNames)
        self.DVGeo.addPointSet(self.leTePts, f"{self.name}_lete", compNames=compNames)

        # Compute the t/c constraints
        self.ToC0 = np.zeros(self.nPoint)

        for i in range(self.nPoint):
            t = geo_utils.norm.eDist(coords[2 * i], coords[2 * i + 1])
            c = geo_utils.norm.eDist(lePt, tePt)

            if self.divideByChord:
                # Divide by the chord that corresponds to this set of constraints
                self.ToC0[i] = t / c
            else:
                self.ToC0[i] = t

        # Compute the absolute t/c at the baseline
        self.max0 = geo_utils.KSfunction.compute(self.ToC0, self.rho)

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coords = self.DVGeo.update(f"{self.name}_coords", config=config)
        self.leTePts = self.DVGeo.update(f"{self.name}_lete", config=config)

        # Compute the t/c constraints
        ToC = np.zeros(self.nPoint)

        for i in range(self.nPoint):
            # Calculate the thickness
            t = geo_utils.norm.eDist(self.coords[2 * i], self.coords[2 * i + 1])
            c = geo_utils.norm.eDist(self.leTePts[0], self.leTePts[1])

            if self.divideByChord:
                # Divide by the chord that corresponds to this constraint section
                ToC[i] = t / c
            else:
                ToC[i] = t

        # Now we want to take the KS max over the toothpicks
        maxToC = geo_utils.KSfunction.compute(ToC, self.rho)

        if self.scaled:
            maxToC /= self.max0

        funcs[self.name] = maxToC

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dToCdCoords = np.zeros((self.nPoint, 2, self.coords.shape[1]))
            dToCdLeTePts = np.zeros((self.nPoint, 2, self.leTePts.shape[1]))

            ToC = np.zeros(self.nPoint)
            for i in range(self.nPoint):
                t = geo_utils.eDist(self.coords[2 * i], self.coords[2 * i + 1])
                c = geo_utils.eDist(self.leTePts[0], self.leTePts[1])

                if self.divideByChord:
                    ToC[i] = t / c
                else:
                    ToC[i] = t

                # Partial derivative of thickness w.r.t coordinates
                p1b, p2b = geo_utils.eDist_b(self.coords[2 * i], self.coords[2 * i + 1])

                # Partial derivative of chord distance w.r.t coordinates
                p3b, p4b = geo_utils.eDist_b(self.leTePts[0], self.leTePts[1])

                if self.divideByChord:
                    # Partial of t/c constraints w.r.t up and down coordinates
                    dToCdCoords[i, 0] = p1b / c
                    dToCdCoords[i, 1] = p2b / c

                    # Partial of t/c constraints w.r.t le and te points
                    dToCdLeTePts[i, 0] = -p3b * t / c**2
                    dToCdLeTePts[i, 1] = -p4b * t / c**2
                else:
                    # Partial of t/c constraints w.r.t up and down coordinates
                    dToCdCoords[i, 0] = p1b
                    dToCdCoords[i, 1] = p2b

            # Get the derivative of the ks function with respect to the t/c constraints
            dKSdToC, _ = geo_utils.KSfunction.derivatives(ToC, self.rho)

            if self.scaled:
                # If scaled divide by the initial t/c value
                dKSdToC /= self.max0

            # Use the chain rule to compute the derivative of KS Max w.r.t the coordinates
            #   - dKSdToC is shape (nPoints), dToCdCoords is shape (nPoints, 2, 3), and dToCdLeTePts is shape (nPoints, 2, 3)
            #   - The final shape of dKSdCoords is (nCoords, 3) and the shape of dKSdLeTePts is always (2, 3)
            # dKSdCoords is the point seeds for the toothpick ends
            dKSdCoords = np.einsum("i,ijk->ijk", dKSdToC, dToCdCoords)
            # reshape this so that the points are stacked
            dKSdCoords = dKSdCoords.reshape(self.nPoint * 2, 3)
            # dKSdLeTePts is the point seeds for the LE/TE points
            dKSdLeTePts = np.einsum("i,ijk->jk", dKSdToC, dToCdLeTePts)

            tmp0 = self.DVGeo.totalSensitivity(dKSdCoords, f"{self.name}_coords", config=config)
            tmp1 = self.DVGeo.totalSensitivity(dKSdLeTePts, f"{self.name}_lete", config=config)

            tmpTotal = {}
            for key in tmp0:
                tmpTotal[key] = tmp0[key] + tmp1[key]

            funcsSens[self.name] = tmpTotal

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """
        handle.write("Zone T=%s\n" % self.name)
        handle.write(
            "Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords) + 2, (len(self.coords) // 2) + 1)
        )
        handle.write("DATAPACKING=POINT\n")

        # Write the coordinates and variables for the toothpicks
        for i in range(len(self.coords)):
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        # Write the coordinates for the chord from LE to TE
        handle.write(f"{self.leTePts[0, 0]:f} {self.leTePts[0, 1]:f} {self.leTePts[0, 2]:f}\n")
        handle.write(f"{self.leTePts[1, 0]:f} {self.leTePts[1, 1]:f} {self.leTePts[1, 2]:f}\n")

        # Write the FE line segment indices for the vertical toothpicks
        for i in range(len(self.coords) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))

        # Write the FE line segment for the chord from LE to TE
        handle.write(f"{len(self.coords)+1} {len(self.coords)+2}\n")


class TESlopeConstraint(GeometricConstraint):
    def __init__(self, name, coords, lower, upper, scaled, scale, DVGeo, addToPyOpt, compNames):
        nCon = (len(coords) - 1) // 2  # Divide the length of the coordinates by 4 to get the number of constraints
        super().__init__(name, nCon, lower, upper, scale, DVGeo, addToPyOpt)

        self.coords = coords
        self.scaled = scaled

        # Embed the coordinates
        self.DVGeo.addPointSet(self.coords, self.name, compNames=compNames)

        # Compute the initial constraints
        self.teSlope0 = self._compute(self.coords, scaled=False)

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates
        self.coords = self.DVGeo.update(self.name, config=config)

        # Compute the constraints
        teSlope = self._compute(self.coords, scaled=self.scaled)

        funcs[self.name] = teSlope

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """
        nDV = self.DVGeo.getNDV()
        step_imag = 1e-40j
        step_real = 1e-40

        if nDV > 0:
            nCoords = self.coords.shape[0]
            dimCoords = self.coords.shape[1]
            dTeSlopePt = np.zeros((self.nCon, nCoords, dimCoords))

            coords = self.coords.astype("D")
            for i in range(nCoords):  # loop over the points
                for j in range(dimCoords):  # loop over coordinates in each point (i.e x,y,z)
                    # perturb each coordinate in the current point
                    coords[i, j] += step_imag

                    # evaluate the constraint
                    conVal = self._compute(coords, scaled=self.scaled)
                    dTeSlopePt[:, i, j] = conVal.imag / step_real

                    # reset the coordinates
                    coords[i, j] -= step_imag

            funcsSens[self.name] = self.DVGeo.totalSensitivity(dTeSlopePt, self.name, config=config)

    def _compute(self, coords, scaled=False):
        """Abstracted method to compute the closeout constraint.

        Parameters
        ----------
        coords : np.ndarray
            The coordinate array.
        scaled : bool, optional
            Whether or not to normalize the constraint, by default False

        Returns
        -------
        np.ndarry
            Array of TE closeout constraints.
        """
        tCoords = coords[:-1]  # exclude the trailing edge in the thickness coords
        top = tCoords[::2]  # Top points of the toothpicks
        bottom = tCoords[1::2]  # Bottom points of the toothpicks
        dVec = top - bottom  # Distance vector between toothpick top/bottom
        t = np.sqrt(np.sum(dVec * dVec, axis=1))  # Complex safe euclidean norm

        xMid = np.zeros((self.nCon + 1, 3))  # nCon + 1 coordinates to account for the TE point

        xMid[: self.nCon] = bottom + dVec / 2  # Midpoints of each thickness con

        xMid[-1] = coords[-1]  # Last point is TE point

        cVec = xMid[:-1] - xMid[1:]  # Chord vectors between each toothpick
        chords = np.sqrt(np.sum(cVec * cVec, axis=1))  # Complex safe euclidean norm
        chords = np.flip(chords)  # Flip the coords so it goes from TE to the first toothpick

        # Take the cumulative sum to get arc length and flip again to match toothpick ordering
        c = np.flip(np.cumsum(chords))

        # Divide the thicknessess by the chord arc lengths
        teSlope = t / c

        if scaled:
            teSlope /= self.teSlope0

        # Return the constraint array
        return teSlope

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle.
        """
        handle.write("Zone T=%s\n" % self.name)
        handle.write(
            "Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coords) - 1, (len(self.coords) - 1) // 2)
        )
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords) - 1):  # Loop over nCoord-1 points (last point is a single TE point)
            handle.write(f"{self.coords[i, 0]:f} {self.coords[i, 1]:f} {self.coords[i, 2]:f}\n")

        for i in range((len(self.coords) - 1) // 2):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))


class ProximityConstraint(GeometricConstraint):
    """
    DVConstraints representation of a set of proximity
    constraints. The user should not have to deal with this
    class directly.
    """

    def __init__(
        self,
        name,
        coordsA,
        coordsB,
        pointSetKwargsA,
        pointSetKwargsB,
        lower,
        upper,
        scaled,
        scale,
        DVGeo,
        addToPyOpt,
        compNames,
    ):
        super().__init__(name, len(coordsA), lower, upper, scale, DVGeo, addToPyOpt)

        self.coordsA = coordsA
        self.coordsB = coordsB
        self.scaled = scaled

        # First thing we can do is embed the coordinates into the DVGeo.
        # ptsets A and B get different kwargs
        self.DVGeo.addPointSet(self.coordsA, f"{self.name}_A", compNames=compNames, **pointSetKwargsA)
        self.DVGeo.addPointSet(self.coordsB, f"{self.name}_B", compNames=compNames, **pointSetKwargsB)

        # Now get the reference lengths
        self.D0 = np.zeros(self.nCon)
        for i in range(self.nCon):
            self.D0[i] = geo_utils.norm.euclideanNorm(self.coordsA[i] - self.coordsB[i])

    def evalFunctions(self, funcs, config):
        """
        Evaluate the functions this object has and place in the funcs dictionary

        Parameters
        ----------
        funcs : dict
            Dictionary to place function values
        """
        # Pull out the most recent set of coordinates:
        self.coordsA = self.DVGeo.update(f"{self.name}_A", config=config)
        self.coordsB = self.DVGeo.update(f"{self.name}_B", config=config)
        D = np.zeros(self.nCon)
        for i in range(self.nCon):
            D[i] = geo_utils.norm.euclideanNorm(self.coordsA[i] - self.coordsB[i])
            if self.scaled:
                D[i] /= self.D0[i]
        funcs[self.name] = D

    def evalFunctionsSens(self, funcsSens, config):
        """
        Evaluate the sensitivity of the functions this object has and
        place in the funcsSens dictionary

        Parameters
        ----------
        funcsSens : dict
            Dictionary to place function values
        """

        nDV = self.DVGeo.getNDV()
        if nDV > 0:
            dTdPtA = np.zeros((self.nCon, self.nCon, 3))
            dTdPtB = np.zeros((self.nCon, self.nCon, 3))

            for i in range(self.nCon):
                pAb, pBb = geo_utils.eDist_b(self.coordsA[i], self.coordsB[i])
                if self.scaled:
                    pAb /= self.D0[i]
                    pBb /= self.D0[i]
                dTdPtA[i, i, :] = pAb
                dTdPtB[i, i, :] = pBb

            funcSensA = self.DVGeo.totalSensitivity(dTdPtA, f"{self.name}_A", config=config)
            funcSensB = self.DVGeo.totalSensitivity(dTdPtB, f"{self.name}_B", config=config)

            funcsSens[self.name] = {}
            for key, value in funcSensA.items():
                funcsSens[self.name][key] = value
            for key, value in funcSensB.items():
                if key in funcsSens[self.name]:
                    funcsSens[self.name][key] += value
                else:
                    funcsSens[self.name][key] = value

    def writeTecplot(self, handle):
        """
        Write the visualization of this set of thickness constraints
        to the open file handle
        """

        handle.write("Zone T=%s\n" % self.name)
        handle.write("Nodes = %d, Elements = %d ZONETYPE=FELINESEG\n" % (len(self.coordsA) * 2, len(self.coords)))
        handle.write("DATAPACKING=POINT\n")
        for i in range(len(self.coords)):
            handle.write(f"{self.coordsA[i, 0]:f} {self.coordsA[i, 1]:f} {self.coordsA[i, 2]:f}\n")
            handle.write(f"{self.coordsB[i, 0]:f} {self.coordsB[i, 1]:f} {self.coordsB[i, 2]:f}\n")

        for i in range(len(self.coords)):
            handle.write("%d %d\n" % (2 * i + 1, 2 * i + 2))
