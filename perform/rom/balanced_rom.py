import time
import os
from math import sin, pi

import numpy as np
from numpy.linalg import svd
from numpy import linalg as LA
import matplotlib.pyplot as plt

from perform.input_funcs import read_input_file

plt.rc("font", family="serif", size="14")
plt.rc("axes", labelsize="x-large")
plt.rc("figure", figsize=(10, 10))


def main():

    rom_input_dir = "../../examples/standing_flame/inputs/brom_inputs"
    romDict = read_input_file("../../examples/standing_flame/inputs/brom_inputs/brom_params.inp")

    BROMOutDir = romDict["BROMOutDir"]
    num_eqs = romDict["num_eqs"]
    nMarkov = romDict["nMarkov"]
    plotHSVs = romDict["plotHSVs"]
    plotEigs = romDict["plotEigs"]

    nMarkov = nMarkov / 2

    if not os.path.isdir(BROMOutDir):
        os.mkdir(BROMOutDir)

    data_file1 = "FOM_Impulse_Resp.npy"
    in_file = os.path.join(rom_input_dir, data_file1)
    impulse_resp = np.load(in_file, allow_pickle=True)
    data_file2 = "steady_init.npy"
    in_file = os.path.join(rom_input_dir, data_file2)
    steady_state = np.load(in_file, allow_pickle=True)

    for varId in range(num_eqs):

        Ctrb = impulse_resp[:, varId, 2:]
        centProf = np.mean(Ctrb[:, : 2 * nMarkov], axis=1, keepdims=True)
        if plotHSVs:
            sigma = HankelSV(Ctrb - centProf, nMarkov)
            if varId == 0:
                sigmap = sigma
            if varId == 1:
                sigmau = sigma
            if varId == 2:
                sigmaT = sigma
            if varId == 3:
                sigmaY = sigma
        Ctrb, normFacProf = normalizeBT(Ctrb[:, : 2 * nMarkov], steady_state[:, varId])

        A, B, C = transform(Ctrb, varId, nMarkov, BROMOutDir, plotEigs)
        TimeAdv(A, B, C, varId, normFacProf, romDict)

    if plotHSVs:
        figFile = os.path.join(BROMOutDir, str(nMarkov) + "HankelSVs.png")
        pltHSVs(sigmap, sigmau, sigmaT, sigmaY, nMarkov, figFile)

    print("BT ROM created!")


def transform(Ctrb, varIdx, nMarkov, BROMOutDir, plotEigs):

    print("Assembling Hankel matrix...")
    H = Ctrb[:, :nMarkov]
    H = np.vstack([H] + [Ctrb[:, i : nMarkov + i] for i in range(1, nMarkov)])
    print("Computing svd(H)...")
    U, sigma, VT = svd(H, full_matrices=False)
    print("Computing balanced ROM matrices...")
    ecum = np.zeros((sigma.shape[0]), dtype=np.float64)
    energyLimit = 0.9999
    for m in range(sigma.shape[0]):
        ecum[m] = np.sum(sigma[: m + 1]) / np.sum(sigma)
        if ecum[m] > energyLimit:
            nBTMode = m + 1
            print("nBTMode=", nBTMode)
            break
    sigma = sigma[:nBTMode]
    U = U[:, :nBTMode]
    VT = VT[:nBTMode, :]
    B = np.diag(np.power(sigma, -0.5)) @ np.transpose(U, (1, 0)) @ H[:, 0]
    B = np.expand_dims(B, axis=1)
    C = H[: Ctrb.shape[0], :] @ np.transpose(VT, (1, 0)) @ np.diag(np.power(sigma, -0.5))
    del H

    Hprime = np.vstack([Ctrb[:, i : nMarkov + i] for i in range(1, nMarkov + 1)])
    Aprime = np.transpose(U, (1, 0)) @ Hprime @ np.transpose(VT, (1, 0))
    del Hprime
    A = np.diag(np.power(sigma, -0.5)) @ Aprime @ np.diag(np.power(sigma, -0.5))

    if plotEigs:
        lambdaVec, V = LA.eig(A)
        pltEigs(lambdaVec, nBTMode, BROMOutDir, varIdx)
        n_unstable = 0
        for nn in range(lambdaVec.shape[0]):
            if abs(lambdaVec[nn]) >= 1.0:
                n_unstable += 1
                print(abs(lambdaVec[nn]))
        print("Number of unstable eigenvalues=", n_unstable)
        print(np.max(abs(lambdaVec)))

    return A, B, C


def TimeAdv(A, B, C, varId, normFacProf, romDict):

    BROMOutDir = romDict["BROMOutDir"]
    pertType = romDict["pert_type"]
    pert_freq = romDict["pert_freq"]
    pert_perc = romDict["pert_perc"]
    press_back = romDict["press_back"]
    num_steps = romDict["num_steps"]
    dt = romDict["dt"]
    sgap = romDict["sgap"]
    plotField = romDict["plotField"]
    rom_vis_interval = romDict["rom_vis_interval"]

    solBT = np.zeros((A.shape[0], 1), dtype=np.float64)
    snapAll = np.zeros((normFacProf.shape[0], num_steps), dtype=np.float64)
    coeff = np.zeros((A.shape[0], num_steps), dtype=np.float64)
    axisrange = np.arange(0, snapAll.shape[0]) * 1e-5

    ERAdt = dt * sgap
    ERATime = 0.0

    print("Time advancement...")

    t0 = time.time()
    for nt in range(num_steps):

        print("Time step=", nt)
        if pertType == "unitImpulse":
            if nt == 0:
                uc = 1
            else:
                uc = 0
        else:
            pressBack = press_back
            pressBack *= pert(ERATime, pert_perc, pert_freq)
            uc = pressBack

        snapBT = C @ solBT
        snapBT = snapBT * normFacProf
        snapAll[:, nt] = np.squeeze(snapBT, axis=1)
        coeff[:, nt] = np.squeeze(solBT, axis=1)

        if pertType == "unitImpulse":
            solBT = A @ solBT + (B * uc)
        else:
            solBT = A @ solBT + (B * uc) * sgap

        if plotField:
            pltField(snapBT, axisrange, nt, varId, rom_vis_interval, BROMOutDir)

        ERATime += ERAdt
    print("online wall-clock time=", time.time() - t0, "sec")


def pert(soltime, pert_perc, pert_freq):
    pert = 0.0
    for f in pert_freq:
        pert += sin(2.0 * pi * pert_freq * soltime)
        pert *= pert_perc

    return pert


def HankelSV(Ctrb, nMarkov):

    H = Ctrb[:, :nMarkov]
    H = np.vstack([H] + [Ctrb[:, i : nMarkov + i] for i in range(1, nMarkov)])
    sigma = svd(H, compute_uv=False)

    return sigma


def normalizeBT(dataArr, solSteady):

    onesProf = np.ones((dataArr.shape[0], 1), dtype=np.float64)

    # normalize by L2 norm sqaured of each variable
    dataArrSq = np.square(dataArr)
    normFacProf = np.sum(np.sum(dataArrSq, axis=0, keepdims=True), axis=1, keepdims=True)
    normFacProf /= dataArr.shape[0] * dataArr.shape[1]
    normFacProf = normFacProf * onesProf

    dataArr = dataArr / normFacProf

    return dataArr, normFacProf


def pltHSVs(sigmap, sigmau, sigmaT, sigmaY, nMarkov, figFile):

    axisrange = np.arange(1, nMarkov + 1)
    axisrange = axisrange.astype(int)
    plt.plot(axisrange, np.log10(sigmap[:nMarkov]), color="black", marker="s", markersize=3, label="Pressure")
    plt.plot(axisrange, np.log10(sigmau[:nMarkov]), color="green", marker="o", markersize=3, label="Velocity")
    plt.plot(axisrange, np.log10(sigmaT[:nMarkov]), color="red", marker="x", markersize=3, label="Temperature")
    plt.plot(axisrange, np.log10(sigmaY[:nMarkov]), color="blue", marker="^", markersize=3, label="Mass Fraction")
    plt.ylabel("log $\sigma_m$")
    plt.xlabel("m")
    plt.grid(which="both", axis="both", color="gray", linestyle="-.", linewidth=0.5)
    plt.legend()

    plt.savefig(figFile)
    plt.close()


def pltEigs(lambdaVec, modeNum, BROMOutDir, varIdx):

    for i in range(modeNum):
        plt.plot([0, lambdaVec[i].real], [0, lambdaVec[i].imag], "bs", markersize=8)
    xlimit = max(np.absolute(lambdaVec.real))
    ylimit = max(np.absolute(lambdaVec.imag))
    plt.xlim((-xlimit - 0.1 * xlimit, xlimit + 0.1 * xlimit))
    plt.ylim((-ylimit - 0.1 * ylimit, ylimit + 0.1 * ylimit))
    plt.ylabel("Imaginary")
    plt.xlabel("Real")
    plt.grid(which="both", axis="both", color="gray", linestyle="-.", linewidth=0.5)
    if varIdx == 0:
        filePath = os.path.join(BROMOutDir, str(modeNum) + "Mode_p_BROMEigs.png")
    elif varIdx == 1:
        filePath = os.path.join(BROMOutDir, str(modeNum) + "Mode_u_BROMEigs.png")
    elif varIdx == 2:
        filePath = os.path.join(BROMOutDir, str(modeNum) + "Mode_T_BROMEigs.png")
    else:
        filePath = os.path.join(BROMOutDir, str(modeNum) + "Mode_Y_BROMEigs.png")
    plt.savefig(filePath)
    plt.close()


def pltField(snapBT, axisrange, nt, varId, rom_vis_interval, BROMOutDir):

    if ((nt) % rom_vis_interval) == 0:
        plt.plot(axisrange, snapBT, color="black", linewidth=4, label="NI-BROM")
        if varId == 0:
            plt.ylabel("Pressure (Pa)")
            field_dir = os.path.join(BROMOutDir, "p")
            figFile = os.path.join(field_dir, "snapshot" + format(nt, "04d") + ".png")
        elif varId == 1:
            plt.ylabel("Velocity (m/s)")
            field_dir = os.path.join(BROMOutDir, "u")
            figFile = os.path.join(field_dir, "snapshot" + format(nt, "04d") + ".png")
        elif varId == 2:
            plt.ylabel("Temperature (K)")
            field_dir = os.path.join(BROMOutDir, "T")
            figFile = os.path.join(field_dir, "snapshot" + format(nt, "04d") + ".png")
        else:
            plt.ylabel("Mass Fraction")
            field_dir = os.path.join(BROMOutDir, "Y")
            figFile = os.path.join(field_dir, "snapshot" + format(nt, "04d") + ".png")
        if not os.path.isdir(field_dir):
            os.mkdir(field_dir)
        plt.xlabel("X (m)")
        plt.legend()
        plt.savefig(figFile)
        plt.close()


if __name__ == "__main__":
    main()
