# A review on the applications of nuclear magnetic resonance (NMR) in the oil and gas industry: laboratory and field-scale measurements

Mahmoud Elsayed \(^{1}\) . Abubakar Isah \(^{1}\) . Moaz Hiba \(^{1}\) . Amjed Hassan \(^{1}\) . Karem Al-Garadi \(^{2}\) . Mohamed Mahmoud \(^{1}\) . Ammar El-Husseiny \(^{1}\) . Ahmed E. Radwan \(^{3}\)

Received: 9 January 2022 / Accepted: 11 February 2022 / Published online: 14 March 2022 © The Author(s) 2022

## Abstract

This review presents the latest update, applications, techniques of the NMR tools in both laboratory and field scales in the oil and gas upstream industry. The applications of NMR in the laboratory scale were thoroughly reviewed and summarized such as porosity, pores size distribution, permeability, saturations, capillary pressure, and wettability. NMR is an emerging tool to evaluate the improved oil recovery techniques, and it was found to be better than the current techniques used for screening, evaluation, and assessment. For example, NMR can define the recovery of oil/gas from the different pore systems in the rocks compared to other macroscopic techniques that only assess the bulk recovery. This manuscript included different applications for the NMR in enhanced oil recovery research. Also, NMR can be used to evaluate the damage potential of drilling, completion, and production fluids laboratory and field scales. Currently, NMR is used to evaluate the emulsion droplet size and its behavior in the pore space in different applications such as enhanced oil recovery, drilling, completion, etc. NMR tools in the laboratory and field scales can be used to assess the unconventional gas resources and NMR showed a very good potential for exploration and production advancement in unconventional gas fields compared to other tools. Field applications of NMR during exploration and drilling such as logging while drilling, geosteering, etc., were reviewed as well. Finally, the future and potential research directions of NMR tools were introduced which include the application of multi- dimensional NMR and the enhancement of the signal- to- noise ratio of the collected data during the logging while drilling operations.

Keywords NMR · Petrophysics · Enhanced oil recovery · Unconventional gas · Geosteering

## Introduction

Understanding the rock- fluid interaction phenomena in porous media is an imperative issue in oil and gas applications both for industrial and academic purposes. Rock and fluid properties should be determined accurately for better characterization and understanding of the reservoir. In field applications, most of these properties can be obtained using suite of logging tools (Ellis and Singer 2007; Liu 2017). However, accurate measurements for important petrophysical properties such as permeability and capillary pressure curves cannot be achieved using the current technology of well logs (Lalanne and Rebelle 2014). Due to this, laboratory- scale measurements are significant to evaluate oil and gas reservoirs and to accurately determine the reserves and potential recovery approaches. Hence, integration between laboratory and field- scale measurements is needed. Low- field NMR offers both options: logging tools run field- scale measurements of fluids inside the formation, while benchtop instrument measurements (laboratory scale) provide access for cross- validation of the field- scale measurements and can be extended for further potential studies (Kleinberg et al. 1990; Mitchell 2016; Song and Kausik 2019). Both laboratory and field- scale measurements could be performed at an operating Larmor frequency of \(\omega = 2\) MHz; however, lab- scale measurements could be performed using higher magnetic field strength allowing for wide range and more diverse applications.

The NMR logging while drilling (LWD) tool is composed of a permanent magnet, magnetic field gradient, and radio- frequency (RF) antenna that excites the formation fluid near the wellbore (Hurlimann and Heaton 2015; Kenyon 1997; Kleinberg and Jackson 2001). Longitudinal relaxation time \((T_{f})\) , transverse relaxation time \((T_{2})\) , fluid diffusion coefficient \((D)\) , and 2D NMR (such as \(T_{f} - T_{2} - T_{2} - D)\) can be acquired using the up- to- date technology that exists in the industry. The depth of investigation could reach tens of centimeters (flushed zone) which could be dominated by the drilling mud filtrate. However, the acquired signal could easily separate the fluid phases allowing trustful interpretation of the NMR logging data as will be discussed in the "4 NMR Applications in Field Scale." section. Furthermore, the data could be acquired in motion or stationery, in cased and open hole, or logging while drilling (LWD). Several NMR logging tools have been invented and developed by service companies such as Schlumberger, Baker Hughes, and Halliburton. In terms of operating Larmor frequency, Schlumberger developed CMR- Plus with highest operating frequency in the industry (2 MHz with minimum echo time \(= 0.2\) ms). They also have MR scanner which reads at multiple depth of investigation up to \(10.2\) cm, in addition to the LWD tool referred as proVISION plus (DePavia et al. 2003; Vij et al. 2018). Halliburton provides the highest depth of investigation (102 cm) tool with operating frequency ranges between 0.547 and 1.183 MHz with minimum echo time \(= 0.2\) ms referred as XMR (Chen et al. 2018a, b; Ge et al. 2021). Furthermore, they developed LWD tool, called MRIL- WD, with lower operating frequency (0.5 MHz with minimum echo time \(= 0.5\) ms). Baker Hughes also invented both well- logging (MR eXplorer) and LWD (MagTrak) with operating frequency ranged between 0.45- 0.88 MHz and 0.35- 0.50 MHz, respectively (Chen et al. 2003; Coman et al. 2014). This variety of options and the broad range of applications make NMR logging tools one of the most widely used tools nowadays. The main components for NMR benchtop basic experiments at the laboratory scale are: static magnetic field \(\mathbf{B}_{0}\) in order to induce polarization through the sample, coil to apply RF radiation at Larmor frequency, and coil to detect emitted oscillating NMR signal; however, usually the two coils are the same (Mitchell et al. 2014a, b, c). More flexibility and diverse applications of NMR measurements are acquired using laboratory- scale experiments (Ouellette et al. 2015). NMR has proven its robustness in the energy industry with the specific examples of lithium ion battery electrolytes (Wiemers- Meyer et al. 2019), chemical structure during in- situ combustion conditions (Pan et al. 2021), upstream (Chen et al. 2018a, b), and downstream oil and gas industry (Abdul Jameel 2021). However, NMR is an immature field of research in the energy industry and opens several potential research projects. Laboratory measurements can be applied using wide range of magnetic field strengths that are tailored to the specifications of the designed experiments (Johns et al. 2015). NMR spectrometers are categorized based on the magnetic field strength and the common experiment conducted in each type as follows: High- field NMR \((\mathbf{B}_{\mathrm{o}} > 3 \mathrm{T})\) , Intermediate- field \((1 \mathrm{T}< \mathbf{B}_{\mathrm{o}}< 3 \mathrm{T})\) , and Low- field \((\mathbf{B}_{\mathrm{o}}< 1 \mathrm{T})\) (Mitchell et al. 2013). High- field NMR are usually used in chemistry (usually used to elucidate molecular and solid structure) that contain cryogenically cooled superconducting components to generate strong magnetic field with high sensitivity (Ladd et al. 2018; Moser et al. 2017; Richardson 1999). This magnet is large in size costing more than \$1,000,000 and requires routine and intensive maintenance. Intermediate- field NMR is commonly implemented for clinical diagnosis as it is suitable for imaging purposes usually referred as magnetic resonance imaging (MRI) (Broche et al. 2019; Gordon et al. 1982; Neuringer 1990; Zia et al. 2019). Several scholars showed that Intermediate field could also be applied for rock core analysis measurements (Almenningen et al. 2020; Borgia et al. 1994; Fordham et al. 1993; Gummerson et al. 1979). The low- field NMR magnet is permanent magnet that does not require cryogens; hence, it has weaker magnetic field. It is much more common in engineering systems and porous media studies (relaxation and diffusion for rock cores). Furthermore, some low- field magnets are portable and small in size (Blümich 2019).

of NMR measurements are acquired using laboratory- scale experiments (Ouellette et al. 2015). NMR has proven its robustness in the energy industry with the specific examples of lithium ion battery electrolytes (Wiemers- Meyer et al. 2019), chemical structure during in- situ combustion conditions (Pan et al. 2021), upstream (Chen et al. 2018a, b), and downstream oil and gas industry (Abdul Jameel 2021). However, NMR is an immature field of research in the energy industry and opens several potential research projects. Laboratory measurements can be applied using wide range of magnetic field strengths that are tailored to the specifications of the designed experiments (Johns et al. 2015). NMR spectrometers are categorized based on the magnetic field strength and the common experiment conducted in each type as follows: High- field NMR \((\mathbf{B}_{\mathrm{o}} > 3 \mathrm{T})\) , Intermediate- field \((1 \mathrm{T}< \mathbf{B}_{\mathrm{o}}< 3 \mathrm{T})\) , and Low- field \((\mathbf{B}_{\mathrm{o}}< 1 \mathrm{T})\) (Mitchell et al. 2013). High- field NMR are usually used in chemistry (usually used to elucidate molecular and solid structure) that contain cryogenically cooled superconducting components to generate strong magnetic field with high sensitivity (Ladd et al. 2018; Moser et al. 2017; Richardson 1999). This magnet is large in size costing more than \$1,000,000 and requires routine and intensive maintenance. Intermediate- field NMR is commonly implemented for clinical diagnosis as it is suitable for imaging purposes usually referred as magnetic resonance imaging (MRI) (Broche et al. 2019; Gordon et al. 1982; Neuringer 1990; Zia et al. 2019). Several scholars showed that Intermediate field could also be applied for rock core analysis measurements (Almenningen et al. 2020; Borgia et al. 1994; Fordham et al. 1993; Gummerson et al. 1979). The low- field NMR magnet is permanent magnet that does not require cryogens; hence, it has weaker magnetic field. It is much more common in engineering systems and porous media studies (relaxation and diffusion for rock cores). Furthermore, some low- field magnets are portable and small in size (Blümich 2019).

Most of the NMR application in petroleum engineering are focused in petrophysics and enhanced oil recovery (EOR) for the laboratory- scale studies. NMR petrophysical core analysis is a non- invasive, powerful, and reliable tool for the routine core analysis (RCA) as it provides accurate determination for porosity, pore size distribution, fluids saturation, and permeability (Arnold et al. 2006; Elsayed et al. 2021a; Freedman et al. 2001; Howard and Kenyon 1992; Mitchell et al. 2015; Mitchell and Fordham 2014). Moreover, special core analysis (SCA) experiments such as wettability, capillary pressure, and clay minerals analysis can be determined using NMR (Al- Garadi et al. 2022; Connolly et al. 2017; Elsayed et al. 2020a; Freedman et al., 2003a; Newgord et al. 2020; Tandon et al. 2020; Tandon and Heidari 2018; Worden and Morad 1999). Numerous EOR methods including acid and fracture stimulation, thermochemical fluids for gas condensate removal, miscible gas injection (huff 'n' puff), \(\mathrm{CO}_{2}\) injection, and thermal- EOR have been evaluated with the use of different NMR measurements (Adebayo et al. 2020a, b; Chen et al. 2018a, b; Dang et al. 2019; Elsayed et al. 2021b, 2020b; Hassan et al. 2020; Mukhametdinova et al. 2020). Drilling and production and engineering operation issues are also addressed in the literature. For example, the evaluation of drilling mud infiltration, and the determination of water- in- oil emulsion droplet size and the evaluation of different demulsifier have been extensively studied (Adebayo and Bageri, 2020a; Salomon Marques et al. 2020).

This article intends to review and cover the applications of NMR in the oil and gas industry including laboratory and field measurements. In terms of laboratory- scale measurements, detailed discussion includes the application of NMR in petrophysics and EOR is presented, in addition, important special topics in drilling and production engineering are also covered. Logging applications including geosteering and logging while drilling (LWD) measurements are highlighted in this article including some field applications. Due to its wide range of application, this paper would serve as a guide for the oil and gas industry in terms of future research studies and field operations.

## NMR theory

The underlying principle of NMR is that some of the nuclei are inherently magnetic. Chemical and isotopic NMR properties are defined by two constants: Nuclear spin quantum number \(I\) and gyromagnetic ratio \((\gamma)\) which is defined as the magnitude of the nuclear magnetic moments (Diehl 2008; Johnson 1999; Levitt 2013; Mazumder and Dubey 2013; Wong 2014). The nucleus is called NMR active if its quantum number \(I\) is larger than zero and has a relatively high gyromagnetic ratio. The following Table 1 shows typical values of some nuclei of \(I, \gamma\) , and their natural abundance:

Table 1 I and typical values for some nuclei

| Nucleus | Spin number (I) | γ (10⁶ rad s⁻¹ T⁻¹) | Natural abundance % |
|---------|------------------|----------------------|---------------------|
| ¹H      | 1/2              | 267.522              | ~99.9               |
| ²H      | 1                | 41.066               | 0.015               |
| ¹²C     | 0                | -                    | 98.93               |
| ¹³C     | 1/2              | 67.283               | 1.1                 |
| ¹⁴N     | 1                | 19.338               | 99.63               |
| ¹⁶O     | 0                | -                    | 99.7                 |
| ¹⁷O     | 5/2              | -36.281              | 0.037               |
| ²³Na    | 3/2              | 70.761               | 100                 |

The concept of NMR is based on applying a magnetic field causing precession to the nuclear spins. The precession frequency (Larmor frequency) is determined by the strength of the magnetic field according to the following equation (Hahn 1950a):

\[\omega = \gamma B_{o} \quad (1)\]

where \(\omega\) is the precession (Larmor) frequency in \((\mathrm{MHz})\) , \(\gamma\) is the gyromagnetic ratio of the nucleus under investigation in \((\mathrm{MHz / T})\) , and \(\mathrm{B}_{0}\) is the strength of the static magnetic field in (T). In order to induce the NMR signal, another magnetic field \((\mathrm{B}_{1})\) in form of radio- frequency pulses is applied in the perpendicular plane (conventionally \(\mathrm{x} - \mathrm{y}\) plane) to the static magnetic field. This will excite the nuclear spin away from their equilibrium state allowing the detection of the nuclear spins precessions in the \(\mathrm{x} - \mathrm{y}\) plane. The decay of the oscillating signal from the NMR experiment is sensitive to molecular dynamics which is called relaxation analysis. It is a powerful and useful probe of adsorption, confinement, and pore characteristic in porous media.

## Relaxation measurements

There are two main forms of NMR relaxation: longitudinal relaxation \((T_{I})\) and transverse relaxation \((T_{2})\) . Longitudinal relaxation \((T_{I})\) , also known as spin- lattice relaxation time, is the process by which the magnetic moment returns to thermal equilibrium, and it occurs during the exponential recovery process as described in the following equation (Vold et al. 1968):

\[\frac{M_{z}(\tau_{1})}{M_{0}} = 1 - 2\exp \left(-\frac{\tau_{1}}{T_{1}}\right) \quad (2)\]

where \(M_{z} / M_{0}\) is the longitudinal magnetization recovery, \(\tau_{1}\) is the delay time (s) and \(T_{I}\) is the longitudinal relaxation time constant (s). To measure \(T_{I}\) relaxation time, inversion recovery (IR) pulse sequence experiment as shown in Fig. 1 is performed. Inversion recovery starts with \(180^{\circ}\) RF pulse that inverts the system magnetization, then the system is allowed to recover longitudinally for a variable time \((\tau_{1})\) , then \(90^{\circ}\) RF pulse flips the magnetization into the transverse plane, hence the signal can be detected. If \(\tau_{1}\) is cycled through, an

[Figure 1: Inversion recovery (IR) pulse sequence]

exponential recovering signal is acquired which can fit Eq. 2 to obtain \(T_{f}\) relaxation time.

Transverse relaxation \((T_{2})\) , also known as spin- spin relaxation time, describes the dephasing of the individual magnetic moments over time in the transverse \(\mathbf{x} - \mathbf{y}\) plane Hahn 1950b). The local magnetic field through samples is inhomogeneous, hence each spin experiences a slightly different precession frequency. The time constant characterizes this process is called effective relaxation time \(T_{2}^{*}\) which is governed by the following equation (Callaghan 1993):

\[\frac{1}{T_2^*} = \frac{1}{T_2} +\gamma \Delta B_0 \quad (3)\]

The second term in the RHS describes static field inhomogeneity of the spectrometer which is an additional source of dephasing. This can be overcome by applying series \(180^{\circ}\) RF pulses to refocus the additional dephasing using CPMG pulse sequences as illustrated in Fig. 2 (Carr and Purcell 1954; Meiboom and Gill 1958). By applying CPMG pulse sequence, the exponential decay signal as described in Eq. 4 decays only due to \(T_{2}\) and the inherent field inhomogeneity is refocused and canceled out.

\[M_{xy}(n_{e}) = \exp \left(-\frac{n_{e}}{T_{2}}\right) \quad (4)\]

where \(M_{xy} / M_{0}\) is the transverse magnetization decay, \(t_{e}\) is the echo time (s), \(n\) is the number of echoes, and \(T_{2}\) is the transverse relaxation time constant (s).

[Figure 2: (CPMG) pulse sequence]

In order to have relaxation time distribution rather than a single time relaxation time constant as shown in Eqs. 2 and 4, inversion of relaxation time data is necessary. The acquired NMR data are generally described by the 1st kind Fredholm integral equation as following (Wilson 1992):

\[\frac{M(t)}{M_0} = \int F(T_{1,2})K_{1,2}d(\log (T_{1,2})) + \epsilon (t) \quad (5)\]

where \(F(T_{1,2})\) is the distribution of relaxation time constant, \(K_{1,2}\) is kernel function describing the expected form of NMR relaxation data (e.g., For \(T_{f}\) measurements, an exponential growth (recovery) is expected whereas \(T_{2}\) measurements is represented by exponential decay), and \(\epsilon (t)\) represents the experimental noise. The NMR data is inverted using Laplace inversion where the data is re- written in a vector- matrix form \(M = K F + e\) ,where \(M\) is the acquired data vector, \(K\) is the kernel matrix, \(F\) is the target probability distribution vector that we need to solve, \(e\) is the noise vector. To obtain F, this term \(\| M - K F\|\) should be minimized such that it satisfies the following:

\[\| M - K F\| < \sigma \quad (6)\]

where \(\sigma\) is the noise variance. Unfortunately, this is an ill- posed problem in the presence of noise, and it has an infinite number of solutions. Hence, some mathematical manipulation is needed in order to obtain a stable and physical solution (Tikhonov and Arsenin 1977). Valid constraints for \(F\) such as non- negative, well- defined range of \(10^{- 4}\) to \(10^{1}\) s, discretized into well- defined number of values, and smooth. In order to smoothen the data, a penalty term is added in the minimization function. Tikhonov regularization uses a penalty function that cancels unwanted fluctuations; hence, we solve:

\[\| M - K F\| +a\| F\| ^2 \quad (7)\]

The level of smoothness is determined by the smoothing parameter \(\alpha\) (Wahba and Wang 1990). An optimization of \(\alpha\) is achieved using robust algorithm called generalized cross- validation (GCV).

## Diffusion measurements

The main difference between relaxation and diffusion measurements is the application of magnetic field gradient (Mitchell 2016; Price 1997; Willis et al. 2016). In pulsed field gradient (PFG) diffusion measurements, an application of magnetic field gradient \((g)\) for a specific time \((\delta)\) is implemented to encode the nuclear spins phase shift from their original position \(r(0)\) . After diffusion time \((\Delta)\) , second magnetic field gradient is implemented for the same duration \((\delta)\) but with the negative value \((- g)\) to decode the phase shift (Hrabe et al. 2007; Mazumder and Dubey 2013; Nicolay et al. 2001). There will be no phase shift if the nuclear spins in the system has not diffused. Otherwise, the nuclear spins will have a net phase shift equal to:

\[\Phi = \gamma \mathrm{g}[\mathrm{r}(\Delta) - r(0)] \quad (8)\]

The PFG NMR signal decay due to diffusion was firstly introduced by Stejskal and Tanner by the following equation (Stejskal and Tanner 1965):

\[\frac{S(g)}{S_0} = \exp (-D(\gamma \delta g)^2\Delta) \quad (9)\]

where \(S(g)\) is the received signal with varying applied magnetic field gradient magnitude, \(S_{0}\) is the signal received in the absence of the applied magnetic field gradient, \(D\) is the self- diffusion coefficient \((\mathrm{m}^2 /\mathrm{s})\) . Figure 3a shows the basic PFG pulse sequence which is usually referred as pulsed gradient spin echo (PGSE) sequence. The PGSE sequences are relaxing via \(T_{2}\) mechanism because the magnetization is stored in the \(\mathbf{x} - \mathbf{y}\) plane; hence, care must be taken while varying diffusion time in such a way that signal could be acquired before relaxation of spins. Usually, \(T_{1}\) is greater than \(T_{2}\) in rocks, and hence it is more convenient to apply

[Figure 3: a Pulsed field gradient spin echo (PGSE) (b) pulsed field gradient stimulated spin echo (PGSTE) (c) 13-interval bipolar alternating pulsed field gradient stimulated echo (APGSTE) pulse sequences representation]

2 pulsed gradient stimulated spin echo (PGSTE). In PGSTE sequence shown in Fig. 3b, the magnetization is stored in the z- axis enabling longer diffusion time to be utilized with Eq. 9 still holds to PGSTE sequence. The magnetic susceptibility difference between rock grain and pore fluids in saturated rocks induces an internal magnetic field gradient that contributed to the diffusion attenuation. (Cots et al. 1989) proposed a new pulse sequence called 13- interval bipolar alternating pulsed gradient stimulated echo (APGSTE) illustrated in Fig. 3c. The APGSTE pulse sequence consists of series of 90 and \(180~\mathrm{RF}\) pulses along with encoding and decoding magnetic field gradient pulses. APGSTE pulse sequence also features gradient that remove any coherent gradient echoes and undesirable coherent spins. This first part of the sequence shown in Fig. 3c (up to the 2nd \(90^{\circ}\) pulse) excites the spin ensemble into the longitudinal axis (z- axis), provide spatial encoding and returns it for storage. The spin ensemble is returned to the transverse plane (x- y plane) by the 3rd \(90^{\circ}\) pulse after the end of the diffusion time and decoded prior the signal acquisition. The signal attenuation as a result of incoherent phase shift during the diffusion time period between encoding and decoding the gradient pulses. This sequence has the capability to minimize the effect of internal gradient which is achieved by the bipolar gradient pairs applied in the encoding and decoding periods. The 2nd magnetic field gradient of each of these pairs is refocused by the 180 pulse and thus operate in the same direction causing a cumulative phase accumulation. For the internal gradient, the imparted phase shift of which due to the period before \(180^{\circ}\) pulse is immediately refocused by the period after the \(180^{\circ}\) pulse and it governs by the following equation:

\[\ln \frac{S}{S_0} = -D_R(2\delta \gamma g)^2 (\Delta -\frac{3}{2}\lambda -\frac{\delta}{6}) \quad (12)\]

The only variable in Eq. 8 is the applied gradient strength, \(g\) , which cause the signal to be attenuated. The signal attenuation, \(S / S_{0}\) , is plotted against the gradient strength along with the temporal parameters to obtain a slope that represents the restricted diffusion coefficient.

NMR diffusion measurements have a variety of applications in core analysis as it is sensitive to both fluid chemistry and pore geometry characteristics. Restricted diffusion measurements are utilized to measure the tortuosity which is inversely proportional to permeability. Additionally, NMR diffusion measurements can be performed along different directions that would also give insights about the permeability anisotropy, fracture preferential directions, and acidizing job efficiency.

## Two-dimensional NMR measurements

It is often the case to correlate the relaxation time constants to obtain system's \(T_{I}\) and \(T_{2}\) almost simultaneously. It is simply combining Inversion recovery followed by CPMG measurements as illustrated in Fig. 4. The inversion recovery component encodes \(T_{I}\) recovery into magnetization of the system and then \(T_{2}\) decay is measured. It is worth measuring because the experimental time is equivalent to 1D inversion recovery measurements. A linear combination of two exponentials in term of \(T_{I}\) and \(T_{2}\) expresses the acquired NMR relaxation signal as following

\[\frac{M(\tau_1,n_t)}{M_0} = \exp \left(-\frac{n_t}{T_2}\right)\left[1 - \exp \left(\frac{\tau_1}{T_1}\right)\right] \quad (13)\]

[Figure 4: \(T_{1} / T_{2}\) correlation pulse sequence (IR-CPMG)]

\(T_{I} - T_{2}\) correlation provides a 2D probability density distribution of the system exhibiting combinations of \(T_{I} - T_{2}\) . The mathematics and data processing to acquire 2D probability distribution is very similar to the 1D that was discussed earlier in "2.1 Relaxation Measurements:" section. The acquired 2D- NMR data are also described by the \(1^{\mathrm{st}}\) kind Fredholm integral equation as following (Venkataramanan et al. 2002):

\[\frac{M(\tau_1,nt_e)}{M_0} = \int F(T_1,T_2)K_1K_2dlog(T_1)dlog(T_2) + \epsilon (\tau_1,nt_e) \quad (14)\]

The vector- matrix minimization in case of 2D probability distribution is computationally intractable. This is because adding additional dimension makes the number of values totally huge (e.g., \(K\) will be \(10^{6} \times 10^{4}\) in size) to be stored. Hence, data compression and computationally efficient processes are required. Singular value decomposition (SVD) is a common mathematical operation used in data compression which eliminates the less important data in the matrix to produce a low- dimensional approximation. In addition, Kernel separability: \(K = K_{1} \otimes K_{2}\) is used to treat the expected relaxation characteristic as two mathematical components since \(T_{1}\) and \(T_{2}\) occur at two separable times (e.g., \(\tau_{1}\) for \(T_{1}\) and \(t_{e}\) for \(T_{2}\) ), then the SVD is performed in these two components (Mitchell et al. 2012a). After that, the resulting minimization problem would be manageable by standard desktop PC.

A very useful 2D NMR correlation measurement correlates transverse relaxation time and diffusion coefficient \((T_{2} - D)\) that can access robust applications such as differentiating oil, gas, and water signals (Mitchell et al. 2014a, b, c; Sun et al. 2018). This is achieved by combining PGSE and CPMG experiments as shown in Fig. 5 where PGSTE is usually applied for bulk fluids, and APGSTE is preferred in case of studying rock core plugs as discussed earlier in "2.2 Diffusion Measurements." In order to ensure eddy current stabilization before CPMG sequence, storage delay time, \(\Delta_{store}\) is incorporated between diffusion encoding and acquisition. In addition, the echo time is kept fixed for all train of echoes in the CPMG sequence. The CPMG echo train could be attached to any diffusion editing pulse sequence where the 2D acquired from applying CPMG and APGSTE could be expressed as:

\[\frac{M(g,nt_e)}{M_0} = \exp \left(-\frac{nt_e}{T_2}\right)\exp \left\{g^2 D\delta^2 \gamma^2 \left[\left(\Delta -\frac{3}{2}\lambda -\frac{\delta}{6}\right)\right]\right\} \quad (15)\]

[Figure 5: \(T_{2}\) -D correlation pulse sequence (APGSTE-CPMG)]

A combination of pair of CPMG echo trains separated by mixing time, \(\Delta_{\mathrm{mixing}}\) , as shown in Fig. 6 provides an application of the diffusive and chemical exchange between two environments (Mitchell et al. 2007; Washburn and Callaghan 2006). When spins diffuse or exchange between two environments featured by two different \(T_{2}\) relaxation times during \(\Delta_{\mathrm{mixing}}\) , off- diagonal peak in \(T_{2} - T_{2}\) maps will be observed. The level of exchange is monitored at different \(\Delta_{\mathrm{mixing}}\) times where the change in the intensity of the peaks reflect the level of exchange. Conventionally, the direct dimension is \(T_{2}^{(1)}\) and the indirect dimension is \(T_{2}^{(2)}\) where the data are only acquired after the second CPMG echo train. The acquired 2D data can be expressed as the following:

\[\frac{M(m_{t_e},n_{t_e})}{M_0} = \exp \left(-\frac{m_{t_e}}{T_2^{(1)}}\right)\exp \left(-\frac{n_{t_e}}{T_2^{(2)}}\right) \quad (16)\]

[Figure 6: \(T_{2} - T_{2}\) correlation pulse sequence (CPMG-CPMG)]

All the 2D NMR data can be processed and inverted using the same technique presented earlier for \(T_{f} - T_{2}\) where Fredholm integral equation of the 1st kind (J Mitchell et al. 2012b).

Table 2 shows a summary of the 2D NMR measurements types and applications. Note that the applications are not limited to the ones in the table, however, they represent the widely used in the literature.

Table 2 2D NMR measurements types and applications

| Measurements | Applications | References |
|--------------|--------------|------------|
| T₁-T₂ maps | Fluid-surface interaction<br>Fluid typing<br>Permeability<br>Wettability<br>Heavy oil reservoir evaluation<br>Surface relaxivity (S/V ratio) | Weber et al. (2009)<br>Fleury and Romero-Sarmiento (2016a)<br>Cheng et al. (2017)<br>Valori and Hursan (2017)<br>Guo et al. (2019)<br>Luo et al. (2015) |
| T₂-D maps | Wettability<br>Fluid saturation<br>Pore coupling<br>Asphaltene deposition | Wang et al. (2018a, b)<br>Sakthivel and Elsayed (2021)<br>Johnson and Schwartz (2014)<br>Shikhov et al. (2018a) |
| T₂-T₂ maps | Diffusion exchange | Mankinen et al. (2020) |

## NMR applications in laboratory scale

NMR relaxation in porous media is primarily defined in terms of biphasic fast exchange between free and adsorbed populations (Godefroy et al. 2001; Korb et al. 1993). The adsorbed surface layer of the pore surface and the free fluids in the center of the pore exchange rapidly based on the observed relaxation represented by the following equation (Turov and Mironyuk 1998; Turov et al. 1997):

\[\frac{1}{T_{1,2}} = \frac{1 - P}{T_{1,2bulk}} +\frac{P}{T_{1,2surface}} \quad (17)\]

\(P\) defines the population of the adsorbed surface layer. The previous equation relates the observed relaxation rate experimentally by weighted average of the bulk and surface relaxation. Relaxation mechanisms within adsorbed surface layer are dipolar in nature due to high molecular density and slow dynamics (Callaghan 2011; Price 2009; Watson and Chang 1997). Moreover, specific dipolar interactions depend on the pore surface chemistry where the reduction in molecular mobility caused by adsorption enhances the relaxation rates (Bloembergen et al. 1948; Gladden and Mitchell 2011; Lin et al. 2006; Huan Wang et al. 2021a, b). While encountering surface bound dipoles, it is important to consider the surface bound hydroxyl solid hydrocarbons such as kerogen, and paramagnetic metal ions (Song and Kausik 2019). These metal ions have unpaired electrons which would dominate the relaxation characteristic due to the very large electron gyromagnetic ratio as compared to the proton gyromagnetic ratio \((\gamma_{e} = 650\gamma_{H})\) (Hoult and Richards 1976; Weil and Bolton 2006). The population of the adsorbed surface layer can be written in the term of the surface- to- volume ratio of the pore structure, \(SV\) as following:

the relaxation characteristic due to the very large electron gyromagnetic ratio as compared to the proton gyromagnetic ratio \((\gamma_{e} = 650\gamma_{H})\) (Hoult and Richards 1976; Weil and Bolton 2006). The population of the adsorbed surface layer can be written in the term of the surface- to- volume ratio of the pore structure, \(SV\) as following:

\[P = \delta \frac{S}{V} \quad (18)\]

where \(\delta\) is the length scale which is representative of the thickness of the adsorbed surface layer. It is the length scale over which surface bound dipole can interact with fluid. This length scale, \(\delta\) , normalized by the \(T_{1,2, surface}\) is well- known as the surface relativity, \(\rho_{2}\) , which is the most important concept in relaxation studies in porous media. Surface relaxivity provides relationship between observed relaxation rate and pore size (Bowers et al. 1995; Ge et al. 2021). By assuming more free bulk fluid than the absorbed in surface layer \((1 - P > > P)\) and large \(SV\) , the approximate expression reveals that the relaxation rate depends on the surface chemistry and material characteristic of the pore structure as following:

\[\frac{1}{T_{1,2}} \approx \frac{1}{T_{1,2bulk}} + \rho_{2}\frac{S}{V} \quad (19)\]

Unlike \(T_{f}\) relaxation, the observed \(T_{2}\) relaxation rate for water saturated porous media would be enhanced in the presence of the induced internal magnetic field gradient (Kenyon et al. 1988). The magnetic susceptibility difference between the solid and fluid phases causes the internal magnetic field. The magnetic susceptibility defines the degree to which a material is magnetized with the static magnetic field (Hu 1998). This could have some complication effects on the acquired signal, e.g., reducing observable signal from inherently short \(T_{2}\) components. The governing equation for \(T_{2}\) relaxation of a fluid confined in porous media in such case can be expressed as (Brownstein and Tarr 1979):

\[\frac{1}{T_2}= \frac{1}{T_{2,bulk}}+ \rho_{2}\frac{S}{V}+ \frac{1}{12}D(\gamma Gt_e)^2 \quad (20)\]

where T2, bulk is the bulk fluid relaxation component, ρ2 is the surface relaxivity constant caused by enhancement of T2 relaxation due to surface interactions; S/V is the surface-to- volume ratio of the pores; D is the confined fluid diffusion coefficient; γ is the gyromagnetic ratio of a proton nuclei (2.675 *10⁸ s⁻¹ T⁻¹ for ¹H nuclei); G is the internal magnetic field gradient; and te is the echo time between the time between the applied 180° pulses. The internal magnetic field term could be minimized by using reducing value of te and performing experiments at low magnetic field.

## Petrophysics

The relaxation time is a function of the fluid in the pores and the characteristics of the porous medium (Arnold et al. 2007). The magnetization (M) and the hydrogen nuclei relaxation of the fluids contained in the pores of a porous medium is measured by the pulsed NMR. The number of hydrogen nuclei is proportional to the magnetization in the magnetic region of the sensor and this can provide the NMR porosity of the media (Mai and Kantzas 2007; Timur 1969). The size distribution of fluid-filled pores can provide an essential information on rock samples, this can be obtained from low- field NMR (Toumelin et al. 2002). (Herlinger and Dos Santos 2018) established that the use of NMR relaxation time (T2) made it possible to determine petrophysical properties including effective porosity, free fluid saturation, irreducible water saturation and other petrophysical properties of rocks. The interpretation of NMR T2 distribution is possible due to the assumption that a relationship exists between pore throat and pore body size measured by NMR. Such assumptions must be set according to the knowledge of fluids and rock properties. Nuclear magnetic resonance interpretation has been improved to be able to derive both drainage and imbibition capillary pressure curve from cores and logs (Gomes 2014; Grattoni et al. 2003) as well as the wetting state of reservoir formation. Hence, NMR serves as a strong mean of petrophysical characterization of reservoir fluids and reservoir rock samples in the laboratory measurements (Mitchell and Fordham 2014). It is worth mentioning that NMR logging tool could also provide information about the porosity, permeability, and reservoir fluid type and saturation around the near wellbore.

### Porosity

Porosity is the primary storage property of the rock (Watson and Chang 1997). Carbonate reservoirs often heterogeneous containing pores of various sizes ranging from small pores in micron or sub-micron to centimeters or even larger vugs (Hidajat et al. 2004). During NMR porosity measurement, the NMR equipment basically detects the hydrogen nucleus contained in the fluid saturating the rock pores. Therefore, when rock is saturated with a single fluid, for instance, water, the detected NMR signal is proportional to the pore volume of the rock. When these pores are spatially close, inter-pore fluid molecules exchange occur between the pore sizes within the relaxation measurements, thus a single peak relaxation time distribution is obtained (Song 2007). The amount of fluid that occupy the pore space of a rock is used to obtain the porosity; the NMR porosity is often computed when a single fluid occupies the pore space. When multiple fluids occupy the pore space, the saturation of each fluid phase can be determined from the knowledge of the amount of that fluid contained in the pores (Watson and Chang 1997). Several factors affecting the accuracy of NMR total porosity measurement of a rock sample including echo spacing (te), magnetic field strength, hydrogen index (HI) of the fluid in the rock pores, repetition time (RT), and rock temperature. Table 3 summarizes different implementation of NMR porosity concept for different lithologies and porous media.

Table 3 Summary of the NMR-porosity studies for different lithologies and porous media

| Authors | Lithology | Remarks |
|---------|-----------|---------|
| Mitchell and Fordham (2014) | 4 Sandstone and 4 Carbonates Outcrop rocks | Using 12 MHz system, T₁-based porosity provides better agreement with the gravimetric porosity (Difference ± 0.5 p.u.) than T₂-based porosity (Difference ± 1.6 p.u.) |
| Tan et al. (2015) | 11 Haynesville Shale reservoir rocks | Compared to gravimetric porosity, NMR porosity is underestimated for shale. NMR porosity model was developed by considering mainly rock fabrics. T₂ indicated that the adsorbed gas contributed more than free gas from the total gas saturation |
| Habina et al. (2017) | Clay Minerals; Smectite, illite–smectite, illite, kaolinite, Fe-Chlorite (Mg-Chamosit | Existence of short NMR T₂ signal (with the center line around 0.1 ms) corresponding to hydroxyl groups. However, for clays consisting of interlayer water, the NMR signal in that range comes from hydroxyls and strongly bound water, and this water remains even after drying at high temperature (up to 200 °C), thus, simply removing signals below 0.1 ms will amount to discrepancies in porosity determination |
| Yan et al. (2018) | 39 tight reservoir rocks (1% < φ < 12%); 10 Sandstone, 5 Carbonates, 24 Shale | Magic-sandwich echo (MSE) pulse sequence provides better estimation than CPMG due to the lower echo time (3 μs) for tight carbonates and sandstone. MSE is not accurate in Shale due to specific relaxation mechanisms (existence of organic matter) |
| Tan et al. (2019) | Igneous rocks | Due to the high content of iron and manganese, some igneous rocks showed 100% relative error in NMR-porosity estimations. They proposed correlation based on the iron and manganese content to have a better estimation |

### Pore size distribution

NMR relaxation time is proportional to the pore size, that is, small pores have smaller values of T2, and large pores have large T2 values. This implies that the change in the NMR relaxation time distribution corresponds to a different pore size within a pores system (Bowers et al. 1995). Hence, this makes it possible to directly relate T2 with a size characteristic of the pore space geometry determined from mercury injection experiments (Brownstein and Tarr 1979; Howard et al. 1993; Kleinberg et al. 1993; Mendelson 1985). Different relaxation time peaks represent the distribution of pore sizes within the rock with each T2 range corresponding to a different pore size.

As the pore space in rock is confined, the T2 relaxation time is assumed fast and has a short te. Thus, T2 of pore fluids cannot be represented by bulk relaxation times. Also, the effect of relaxation time due to diffusion in Eq. 20 can be neglected (Coates et al. 1997; Mitchell et al. 2013). If the surface relaxivity is relatively large, the relaxation time related to the surface dominates the relaxation (Howard et al. 1995). It follows that

\[\frac{1}{T_2}\approx \rho_2^2 \left(\frac{S}{V}\right) \quad (21)\]

where, \(\rho_{2}\) is the surface relaxivity, and \(S / V\) is the surface area of pore- volume ratio. Accordingly, the NMR signal provides relative pore size distribution. It follows that surface relaxivity is a key parameter in determination of pore size distribution (PSD) (Ge et al. 2021; Jaeger et al. 2009; P. Zhao et al. 2020a, b). When a porous medium is saturated with single fluid, each pore size has a characteristic relaxation time decay constant. The smaller the pores the faster the relaxation (short \(T_{2}\) ). Assuming spherical pores, and \(T_{2\mathrm{surface}} < T_{2\mathrm{bulk}}\) , the PSD can be estimated using Eq. 22 (Lawal et al. 2020; Luo et al. 2015; Valori and Nicot 2019).

\[r = 3\rho_{2}T_{2} \quad (22)\]

where \(r^{\prime}\) is the pore radius measured in \(\mu \mathrm{m}\) . Assuming a cylindrical pore shape, the PSD can be estimated as:

\[r = \rho_{2}T_{2} \quad (23)\]

[Figure 7: Comparison of NMR and MICP pore size distributions: Comparison of PSD using NMR shown in blue and PSD derived from mercury porosimetry, shown as orange curve, adapted from (Venkataramanan et al. 2014)]

NMR \(T_{2}\) provides measurements over a wide range of resolution, for instance, microscale (100- 10 μm), mesoscale (10- 100 μm), macroscale (100- 1 mm), and core- scale (1- 100 mm). Several applications of NMR at various scales of resolution includes kerogen content detection (microscale), pore size distribution and permeability (microscale and macroscale), and fluid saturation (micro- to core- scale). Based on pore throat sizes, a rock can be identified as having micropores, mesopores, macropores, vugs, or fractures. Since different pore sizes have different relaxation times, and this relaxation time is proportional to the sizes of the pores, the pore size distribution can be obtained from the \(T_{2}\) relaxation time distribution (Pires et al. 2019): Pores with NMR \(T_{2}\) relaxation time greater than 1000 ms are considered as vugs, while pores with \(T_{2}\) relaxation time between 10 and 100 ms are regarded as smaller macropores, mesopores are described as having \(T_{2}\) relaxation time ranging between 1 and 10 ms, while the micropores are considered to have \(T_{2}\) relaxation time less than 1 ms (Cai et al. 2013; Pires et al. 2019; Radwan et al. 2021).

For pores sizes less than \(1\mu \mathrm{m}\) \(T_{2}\) distribution does not strictly represents PSD for clay minerals because of pore diffusive coupling where pores size is less than the diffusion length (Fleury and Romero- Sarmiento, 2016). Diffusion pore coupling may also occur in multi- modal pore system, such as macro- and micropores in carbonates. In this case, the knowledge of the diffusion coupling effects is necessary to accurately estimate pore sizes. Furthermore, as discussed above, NMR relaxation time is proportional to the size the pores, that is, smaller pores are characterized by shorter relaxation times compared to large pores. Therefore, the NMR signal of water saturated porous media is directly related to the pore size distribution and porosity of the rock. However, for rocks with porosities lower than about \(5\%\) this relationship is somewhat complex (Blumich et al. 2004). Another major drawback in using NMR relaxation times to obtain pore size distributions is insufficient of knowledge of the surface relaxivity of the rock. The traditional practice is to assume that the surface relaxivity is constant for a particular sample, though some researchers have allowed it to vary within a sample (Arns et al. 2006). Extensive efforts were performed to overlap the MICP and NMR relaxation distribution accurately by imaging the rocks using \(\mu \mathrm{CT}\) scanner for better evaluation of surface relaxivity (Benavides et al. 2020; Connolly et al. 2019; Luo et al. 2015).

### Permeability

Permeability is the ease at which fluid flows through a porous medium, and it is one of the fundamental properties of porous media. The basic principle of NMR application in rock permeability determination is relationship that exist between NMR relaxation times and the pore geometry. It is considered as an important tool for the estimation of permeability both in the laboratory and field scales. NMR has reasonably been employed to estimate rock permeability since NMR relaxation times are related to the pore size and geometry. NMR measurements can be implemented in downhole logging tool which provide an opportunity to determine in- situ formation permeability estimation (Watson and Chang 1997). Several correlations and models exist for permeability estimation using NMR, (Seevers 1966) presented a correlation for permeability estimation of fluid in porous media by combining Kozeny equation for permeability and NMR relaxation time. Several authors employed NMR relaxation times and porosity to estimate permeability (Banavar and Schwartz 1987; Kenyon et al. 1988; Rezaee et al. 2012) and found that:

\[k\propto \phi^{a}T_{1}^{b} \quad (24)\]

where \(\phi\) is the porosity of the porous medium, and \(a\) and \(b\) are empirical constants.

Using similar principle, a better permeability estimation was presented by (Timur 1969) where he assumed all pores contribute to fluid transport based on their surface- to- volume ratio \((SV)\) . It should be noted that NMR does not measure fluid flow, it measures static petrophysical properties such as porosity and tortuosity/pore connectivity (Yang et al. 2019) that can be linked to permeability, thus, it is worth noting that permeability cannot be measured by NMR but calculated from NMR measurement using various models. Timur model for permeability is given as:

\[k_{NMR} = \left(\frac{\phi_{NMR}}{C}\right)^m\times \left(\frac{BVM}{BVI}\right)^n \quad (25)\]

Coates et al., (1991) presented for NMR- derived permeability model based on pore surface area commonly referred to as free fluid index (FFI) model that is, the rock fraction that represent the free moveable water in a porous medium given as:

\[k_{NMR} = 10^{-11}\phi^4\times \left(\frac{FFI}{BVI}\right)^2 \quad (26)\]

where BVI is the bulk volume irreducible (non- producible fluids), and BVM is the bulk volume movable (free fluids) and the ratio BVI/BVM a measure for the specific internal pore surface \(\mathrm{S}_{\mathrm{por}}\) C is an empirical parameter. Another commonly used NMR permeability estimation model is the Schlumberger Doll Research (SDR) model (Hidajat et al. 2004):

\[k_{NMR} = aT_{2LM}^{2}\phi^{4} \quad (27)\]

Several models for estimating permeability have been developed (Peng et al. 2019; Solatpour and Kantzas 2019). However, these correlations assumed the same surface relaxivity for the rock samples, thus, there is need to correct for the variations of surface relaxivity across samples. Further, most of the permeability models are based on empirical considerations, and are based on porosity- permeability relation, sometimes the permeability and porosity does not correlate (we may have high porosity but very low permeability). Additionally, these permeability estimation models are also influenced by the wettability of the porous medium (Ji et al. 2020). Lastly, an additional factor/parameter may be required to account for pore throat size and pore connectivity. To increase accuracy of \(k_{NMR}\) , the model parameters (e.g., \(m\) and \(n\) ) in Eq. 25 should be calibrated to local reservoir data, also, the NMR \(T_{2}\) cutoff of the pore throat should be considered rather than the \(T_{2,LM}\) from the NMR pore size distribution (Elsayed et al. 2021a).

### Fluid saturation

The NMR \(T_{2}\) spectrum usually shows a highest amplitude when fully saturated (Isah et al. 2021a, 2021b), also, larger pores are associated with a longer relaxation time. In the case of air- water displacement, as the pores are emptied, the initial water- filled pores are being replaced with air, the amplitude decreases. This is because the disappearance of the longer component of the NMR relaxation time spectrum corresponds directly to the loss of hydrogen containing fluid from the pores due to desaturation (Isah et al. 2021a). This is because NMR relaxation time is sensitive to fluid saturation in the rock pores (Azizoglu et al. 2020). At partial saturation, Eq. 29 implies that \(T_{2}\) will be shorter because \(S / V\) has increased; surface area is the same, but the volume of water has decreased (Howard et al. 1995). If the displacing fluid is air, it replaces the water and generates no NMR signal since it contains no hydrogen (Coates et al. 1997; Howard et al. 1995; Toumelin et al. 2002).

\[T_{2}\approx \frac{S_{w}}{\rho_{2}}\left(\frac{V}{S}\right) \quad (29)\]

where \(S_{w}\) is the fluid saturation, \(S\) is the surface area, \(V\) is the volume of the saturating fluid in the pores (pore volume) and \(\rho_{2}\) is the surface relaxivity which characterizes the relaxation rate on \(S\) . The total amount of fluids in a rock sample is indicated by the initial amplitude of the recorded echo signal which can be used to determine the fluid saturation and porosity using relevant equations. (Isah et al. 2021a) employed this principle to generate NMR brine saturation for both sandstones and carbonates. The distributions of the relaxation times \((T_{f}\) and \(T_{2}\) ) of hydrogen nuclei in rock pores are functions of the size of pores, the fluids present in the pores, and the rock minerals in the pore walls (Kleinberg et al. 1994). (Habina et al. 2017) employed \(T_{f} - T_{2}\) maps to determine fluid distribution in rocks. Fluid distribution and proton mobility of fluids in rock pores can be assessed by employing 2D maps of NMR relaxation times \((T_{f} - T_{2}\) map) in combination with \(T_{f} - T_{2}\) ratio. This is because, the signal from \(T_{f} - T_{2}\) maps corresponds to hydrogens from movable liquid fraction in the pores. These measurements have been useful in the laboratory investigations of both conventional and unconventional rocks (Habina et al. 2017; Timni et al. 2015).

### Capillary pressure

When a rock pore of capillary size is saturated with two or more immiscible fluids, capillary pressure \(P_{c}\) exists. Capillary pressure is the difference in pressure between two immiscible fluids across a curved interface at pressure equilibrium (Isah et al. 2021a). Capillary pressure curves can be estimated using NMR experiments (Norgaard et al. 1999). It can be derived from NMR \(T_{2}\) distribution through the use of local laboratory calibration to obtain a scaling factor (Glorioso et al. 2003; Raheem et al. 2017). A novel method of acquiring the capillary pressure curve using NMR measurement coupled with centrifugation was presented by (Chen and Balcom 2008, 2005). The method involves saturating the core plug and measuring the saturation profile using an NMR when fully saturated. The core plug is then centrifuged once, and the saturation profile is then measured again using the NMR equipment. The method is said to be fast and accurate when compared with the conventional capillary pressure techniques (Faurissoux et al. 2018; Green et al. 2008; Sylta 2010). (Liaw et al. 1996) used fluid saturation distributions obtained by NMR to estimate the relative permeability and capillary pressure. A fairly good agreement between measured and NMR capillary profile was reported. (Baldwin and Spinner 1998; Chen and Balcom 2005) obtained the \(P_{c}\) curves for both drainage and imbibition using NMR. The authors used liquid and frozen hydrocarbon fluids with both \(\mathrm{H}_{2}\mathrm{O}\) and Deuterium oxide \((\mathrm{D}_{2}\mathrm{O})\) . They noted that freezing the fluid prevent saturation redistribution during measurements after centrifugation. (Faurissoux et al. 2018) proposed a method to measure both saturation exponent and capillary pressure using a combined centrifugation and NMR imaging and resistivity profiling. The method is fast and can be employed for both homogeneous and heterogeneous rock samples. Recently, (Isah et al. 2021a) presented a new method to obtain the capillary pressure curves in bimodal rock samples using NMR. They validated their results by comparing with measurements obtained using conventional method. Several NMR- based capillary pressure curves have been presented (Eslami et al. 2013; Hosseinzadeh et al. 2020; Wang et al. 2019; Wu et al. 2021; Xiao et al. 2016). However, it should be noted that, when comparing capillary pressure and NMR relaxation measurements, NMR gives surface - volume ratio of the pores while capillary pressure is related to pore throat size. The two measurements are only precisely comparable if the pore systems are assumed to approach that of a bundle of tubes. That is, when the radius of the pore body is equal to the radius of pore throat where most of sandstone are considered in this category.

### Wettability

The sensitivity of NMR measurements to rock- fluid interaction is well known; wetting fluids contacting rock surface show different NMR responses from that of non- wetting fluids (Valori and Nicot 2019). For instance, \(T_{2}\) relaxation times are enhanced when a fluid comes into contact with rock surface compared to bulk relaxation of the same fluid (Brown and Fatt 1956; Howard 1998). The degree of enhancement depends on the different parameters including area coated by the fluid and strength of rock- fluid interaction (Valori and Nicot 2019). Preceding studies exploited this fact to qualitatively evaluate rock wettability (Al- Mahrooqi et al. 2003; Freedman et al., 2003; Freedman and Heaton 2004; Guan et al. 2002; Zhang et al. 2000) or monitor wettability alteration during different processes such as asphaltene adsorption and deposition (Gonzalez et al. 2016; Shikhov et al. 2019, 2018b). Table 4 presents a summary of the various NMR \(T_{2}\) wettability indices.

\(T_{f} / T_{2}\) ratio performs better than \(T_{2}\) relaxation in wettability prediction when diffusion relaxation is significant (Katika et al. 2017). Therefore, some studies used \(T_{f} / T_{2}\) ratio for assessing wettability qualitatively (Katika et al. 2017; Valori et al. 2017; J. Wang et al. 2018a, b). The main principle behind wettability evaluation from \(T_{f} / T_{2}\) ratio is that molecules of bulk non- viscous fluid exhibit fast and isotropic motion resulting in \(T_{f} / T_{2}\) equal to 1. On the other hand, as the motion of molecules becomes slow or anisotropic such as in high viscous or wetting fluids, \(T_{f} / T_{2}\) tends to be greater than 1 (Valori et al. 2017). Yet, a major challenge for applying this technique occurs when using heavy oil especially containing asphaltene such that the \(T_{f} / T_{2}\) ratio may be greater than 1 influencing wettability interpretation (Valori et al. 2017; Valori and Nicot 2019).

\(T_{2} - D\) NMR measurements also produced excellent wettability predictions (Flaum et al. 2005; Liang et al. 2019; Minh et al. 2015). The main advantage for \(T_{2} - D\) technique is that the fluids separation is improved significantly compared to that when using \(T_{2}\) relaxation approach alone. Thus, calculated effective surface relaxivities of fluids become more accurate (Minh et al. 2015). Finally, NMR has a great potential to be used for in- situ wettability evaluation (Valori et al. 2018). Also, some studies implemented simulation tools to investigate the response of NMR in rock cores of multiphase conditions (Al- Muthana et al. 2012; Looyestijn 2008; Mohnke et al. 2015; Wang et al. 2018a, b).

## Enhanced oil recovery applications

Nuclear magnetic resonance (NMR) presents an effective tool to characterize different operations such as drilling and enhanced oil recovery (EOR) operations (Kenyon 1997). Different types of NMR measurements can be utilized to evaluate the EOR treatments. The common approach is \(T_{2}\) distribution, which can be used to capture the changes in rock porosity system due to EOR treatment. Also, \(T_{2}\) can be used to monitor the fluids' saturations during EOR experiments, which will help in a better design and evaluation for EOR methods. \(T_{2}\) approach can be used to assess the oil saturations at different times, help in monitoring the remaining oil saturation for different EOR techniques in the reservoir (using NMR logging) and in the laboratory experiments. Moreover, the NMR pulsed field gradient (PFG) technique is non- invasive which can be utilized for determining the diffusion coefficient for several fluids can be used at early stages of EOR treatment in order to select the suitable method of EOR based on the pore network. Diffusion measurements can be used to assess the pore coupling, which can help in screening different chemicals before using in the EOR treatment. For example, in poor reservoir connectivity, chemicals that can improve the pores' connectivity can be used while viscous fluids that would require high injection pressure should be excluded. Generally, obtaining the apparent diffusion coefficient can help in mapping the oil and water distributions in porous media, which can be very useful in designing and screening several fluid systems for EOR applications.

NMR can be used to assess the oil saturations at different times, help in monitoring the remaining oil saturation for different EOR techniques in the reservoir (using NMR logging) and in the laboratory (Allsopp et al. 2001; Bryan et al. 2006a, b; Bryan et al. 2006a, b; Goodarzi et al. 2005). Low- field NMR measurements can be used in the laboratory measurements to evaluate the EOR treatments for various types of conventional reservoirs (such as light and heavy oils) and unconventional reservoirs (such as shale oils) (Dong et al., 2020; Markovic et al. 2020). The primary objective of NMR measurements is to screen different chemicals for the EOR applications, such as \(\mathrm{CO}_{2}\) , surfactant, and polymer flooding (Arora et al. 2010; Mitchell et al. 2012c; Mitchell et al. 2012b, 2012a; Suekane et al. 2009). Typically, the distribution of \(T_{2}\) relaxation time is used to provide the saturation profile as a function of distance or along with the treated rock samples. Obtaining the \(T_{2}\) relaxation time profiles before and after the treatment can help in quantifying the remaining oil saturation as well as providing good information about the saturation distribution (Mitchell et al. 2012c). Different techniques can be used to determine the oil saturation including chemicals selective imaging, complete signal suppression, and paramagnetic doping (Baldwin and Yamanashi 1989; Borgia et al. 1994; Brautaset et al. 2008; Doughty and Tomutsa 1996; Enwere and Archer 1992; Green et al. 2008; Norgaard et al. 1999). The selection of the appropriate technique depends on several factors such as rock type, composition, and chemical type (Mitchell et al. 2013).

One of the main advantages of using NMR for evaluating EOR operations is that NMR can provide detailed and continuous monitoring of the remaining oil saturation, which can help in optimizing the EOR performance. Also, the NMR technique can be used to monitor the surfactant progress during the chemical enhanced oil recovery (cEOR) processes, by evaluating the changes in pore surface wettability that occurred due to the surfactant adsorption (Wang et al. 2018a, b). On the other hand, the main limitation for benchtop NMR instruments is that only small core samples should be used (between 2 and 4 inches) which can restrict the analysis of long core samples. Most of the coreflooding experiments are conducted using long core samples (6-20 inches) to avoid the capillary end effect and provide more reliable results. Unfortunately, these long samples cannot be used for NMR measurements, and the available solution is to cut the core samples into small pieces and use composite cores during the chemical flooding (Mitchell et al. 2014a) (Mitchell et al. 2014).

Mitchell et al. (2014b) studied the oil recovery from carbonate rocks during alkaline surfactant (AS) flooding. NMR measurements were used to monitor the oil saturation using spatial \(T_{2}\) profiles. Co- injection of brine and AS was applied, and reservoir conditions of high pressure and high temperature were implemented. The NMR tool was utilized to track the oil- AS interface, which help in understanding the oil recovery mechanisms for different surfactants. Also, the NMR analysis help in assessing the injectivity problems during the AS flooding, by locating the plugged pores where the oil can be displaced by the injected chemicals. (Al Harbi et al. 2017) used advanced NMR techniques to study the performance of nano- surfactant (NS) in recovering the oil from carbonate reservoirs. A Lowfield NMR tool was used to monitor the oil saturation during water flooding and sulfonate- based NS injection, using eight core samples of high permeability (120- 1200 mD). The main objective of using the NMR technique was to understand the primary oil recovery mechanisms. Different injection schemes were applied, and various soaking periods were used. Injection of nano- surfactant followed by water flooding provided the highest oil recovery. The NMR analysis revealed that NS injection can increase oil production by mobilizing the trapped and adsorbed oil from the rock surface.

Dong et al., (2020b) characterized the hydrocarbon flow and the pore structures of tight sands during EOR operations using the NMR technique. The oil recovery during \(\mathrm{CO}_{2}\) injection was determined, and different soaking periods during huff- n- puff (HnP) were applied to optimize the oil production. NMR technique was used to provide the profiles of free fluid (FF), capillary bound fluid (CAF), and claybound fluid (CBF), based on the \(T_{2}\) relaxation time. The fluid displacements across different pore sizes and at different system pressures were analyzed using low- field NMR experiments. The NMR results showed that the free fluid and most of the capillary bound fluid were recovered from the small and medium pores during the first and second cycle of \(\mathrm{CO}_{2}\) injection, while no changes were observed for the clay- bound fluid. Overall, the flooding and NMR measurements helped in optimizing the \(\mathrm{CO}_{2}\) injection by providing the saturation profiles for the free, capillary bound, and claybound fluids.

Moreover, EOR operations can lead to several alterations in the reservoir pores system due to the interactions between the injected fluids and reservoir matrix. More changes were anticipated in the tight reservoirs. Mamoudou et al. (2021) studied the changes in shale microstructure during EOR Huff- n- Puff operations. Crushed samples from Montney and Duvernay shales were used, and a mixture of methane and ethane with different ratios was utilized. NMR technique was integrated with mercury injection capillary pressure (MICP) and isothermal adsorption analysis. The NMR studies showed that the pore throat size was increased by 3- 10 nm due to the EOR treatment, leading to an increase in the rock pore surface area with a factor of 2 on average. The adsorption and SEM analyses confirm the increase of pore surface area. Overall, the NMR analysis showed that gas injection into shale rocks can lead to measurable changes in the shale microstructure, which should be considered during the design and application of EOR operation in tight reservoirs.

Zhu et al. (2021) presented a systematic review on the application of the NMR technique for characterizing the reservoir system during polymer flooding. NMR can be applied at different stages during the polymer flooding; including polymer characterization, polymer movement within the porous media, and polymer plugging. The changes in polymer chemical structure can be assessed using NMR analysis. Also, the gelation parameters such as gel strength and gelation time can be evaluated in- situ by conducting NMR measurements. The main advantage of using the NMR technique for studying polymer flooding operations is that NMR is a non- destructive test that allows for quick and reliable measurements (Al- muntasheri 2008; Wang et al. 2021a, b). Overall, the research in the NMR application for EOR operations is still in its infancy, and limited studies have been conducted. Also, the interaction between oil and injected fluids during polymer flooding is not well understood, and more research is needed in this area. NMR technique will play a significant role in disclosing the rocks and fluids behaviors during EOR operations.

Moreover, on the reservoir scale, NMR logging is used to determine the in- situ oil saturation (Kwak et al. 2017; Zhao et al. 2020a, b). NMR can provide continuous profiles of the oil saturation due to the high acquisition speed of the NMR tool (Mitchell and Fordham 2014). Also, NMR logs are used to monitor the progress of injected chemicals and provide distribution profiles for the chemical and oil saturations (Afsahi and Kantzas 2007). Therefore, it is usually recommended to run NMR logging for all pilot tests of chemical EOR operations. Also, integrating the NMR technique with coreflooding experiments will provide an enhanced assessment of the remaining oil saturation that can be complemented with the field pilots, leading to better oil recovery operations.

## Special topics

### Evaluation of mud filtrate invasion characteristic

Different studies have used NMR technology to assess formation damage induced by drilling fluid invasion in addition to mitigation or removal techniques. In addition, NMR technology has also been used to characterize mud cake formed by drilling fluids and determine its properties. (Bageri et al. 2019) exploited NMR \(T_{2}\) and spatial \(T_{2}\) measurements to assess secondary formation damage induced by the invasion of chelated barite. Chelated barite is formed during the process of removing filter cake, which is formed by barite- based drilling mud, using chelating agents. By injection chelated barite into sandstone and carbonate rocks, they found that barite precipitates are discharged from barium- saturated chelating agent into the rock's pores, and after that, the chelate absorbs cations from the rock minerals. Figure 8 shows the \(T_{2}\) distribution for a very high permeability (211 mD) sandstone sample (A) used in the study. It is obvious that the number of pores with \(T_{2}\) values ranges from 0.8 to \(100~\mathrm{ms}\) experienced a noteworthy reduction. However, some macro pores became larger such that the \(T_{2}\) range expanded from \(300\) to \(900~\mathrm{ms}\) (Gamal et al. 2021) followed the same NMR approach to investigate mud filtrate's effect on the pore system of various sandstone rocks.

[Figure 8: \(T_{2}\) distribution for sandstone sample A before (red) and after (black) barite-laden chelate injection, adapted from (Bageri et al. 2019)]

Adebayo and Bageri, (2020b) introduced a simple NMR method to assess properties of filter cake which includes porosity, thickness, and pore volume. They suggested that the NMR- based technique has an advantage in that it evaluates most of the filter cake properties compare to other methods that can determine only one or two parameters. The method is based on conducting NMR \(T_{2}\) relaxation measurements on rock samples at three stages which are, fully brine saturation, post mud invasion (filter cake is formed), and after mud cake elimination (Adebayo and Bageri, 2020b). The difference in areas under \(T_{2}\) distribution and \(T_{2}\) cumulative porosity curves are used to calculate filter cake properties as was shown in a recent publication (Bageri et al., 2021a). They applied this approach to characterize filter cakes, mainly mud cake porosity and infiltrated solids; the study also compares NMR results with gravimetric and micro- CT measurements. They conducted measurements on 4 consolidated carbonate and sandstone core samples before and after the formation of filter cake. The authors reported that the NMR results correlated excellently with the measured cake's thickness. They determined the mud cake porosity as the difference in porosities obtained from the \(T_{2}\) measurements when mud cake forms and after its removal (Fig. 9a). Similarly, the infiltrated solids porosity that invaded the core samples was obtained (Fig. 9b).

[Figure 9: a \(\mathrm{T}_2\) cumulative porosity of core sample 1 after the invasion with filter cake (black) and after removing the filter cake (red); b \(\mathrm{T}_2\) cumulative porosity of core sample 1 before (red) and after the invasion (black) adapted from (Bageri et al., 2021b)]

Adebayo et al., (2020a, b) extended their preceding works (Adebayo and Bageri, 2020b); they attempted to correlate NMR \(T_{2}\) measurements with mud cake's thickness and porosity in addition to formation damage induced by drilling mud. Various rock types of wide permeability and porosity range were used, and the \(T_{2}\) measurements were conducted at three depth of investigations (DOI) in order to simulate in- situ NMR logging during drilling. The three covered regions are the mud cake, invaded zone, and virgin zone. The difference in areas under \(T_{2}\) curves obtained at these regions correlated well with the mud cake porosity and thickness in addition to formation damage.

Another study by Wu et al. (2019) combined NMR measurement and large- sized model flow experiment to get insights into formation damage induced by fresh water- based drilling mud invasion in sandstone reservoir. A large- sized sandstone formation module with radial depth of \(55.9~\mathrm{cm}\) and thickness of \(10~\mathrm{cm}\) was used; the main purpose of using the large size model, instead of traditional core plugs, is to allow for creating a relative larger damage depth in a long- term invasion of drilling fluid which is a better manifestation of actual in- situ conditions. They used NMR to determine \(T_{2}\) distributions, porosity, and movable saturation of the formation modules at different radial depths. The found a good positive correlation between the formation damage, in terms of permeability, and the decrease ratio of movable water calculated based on NMR \(T_{2}\) distribution as following:

\[MW_{DR}^{m} = \frac{S_{i} - S_{n}}{S_{i}}\times 100\% \quad (30)\]

[Figure 10: Relationship between movable water's decrease ratio and permeability damage rate, adapted after (Wu et al. 2019)]

where \(MW_{DR}^{n}\) is the movable water's decrease ratio at n radial depth of the formation module; \(S_{i}\) and \(S_{n}\) are the initial movable water saturation and the movable water saturation (at n radial depth) of the formation module, respectively. For example, as movable water's decrease ratio changes, the permeability reduction rate of the lower permeability sandstone reservoir is faster than that of a higher permeability as shown in Fig. 10.

Sun et al., (2021) established a morphological correction model of NMR \(T_{2}\) spectrum in oil- base mud invaded zone by comparing the difference of NMR \(T_{2}\) response in oil- based mud and water- based mud wells; This is crucial for accurate interpretation and petrophysical evaluation during NMR logging. Two oil- based mud and three water- based mud wells, in the Xihu Sag sandstone gas formations in the East China Sea basin, were used as a source of the logging data. They found that the estimated NMR permeability (calculated using SDR model) after correcting \(T_{2}\) spectrum morphology is the most accurate compared to NMR permeability before correction and pore fitting permeability.

Several depths of investigations (DOIs) can be achieved by the existing \(T_{2}\) NMR logging tools (Knight et al., 2016). In addition, Magnetic Resonance Imaging Log (MRIL) of some oil companies has well- defined sensitive volume which is typically not affected by mudcake and borehole (Xie et al. 2008). For example, NMR logging technology has the ability to reach several depths of investigation that ranges between 1 and 4 inches from the wall of the borehole into a formation rock. However, mudcake components and properties could limit the capability of NMR. For example, salinity of the formed mudcake could result in a loss of radio- frequency (RF) energy and a reduction in the signal to noise ratio (SNR) (Xie et al., 2008). In addition, \(T_{2}\) relaxation times are significantly reduced when paramagnetic materials, or clay- bound water are present which cause some NMR signals having very small \(T_{2}\) times not to be detected (Xie et al. 2008). Components of filter cake produced by oil- based mud could also lead to failure in NMR interpretation (Sun et al. 2021).

### Emulsion droplet size determination using PFG NMR

Emulsions are dispersion of oil in water or water in oil, it can be more complicated structure (Wong et al. 2015). In oilfield, the emulsion of interest is water in oil (droplets of water dispersed in oil) which is more difficult to separate (Walstra 1993). PFG NMR is a powerful tool to measure the molecular self- diffusion coefficient. The essence of using PFG NMR to measure the emulsion droplet size is by measuring the restricted diffusion caused by the continuous phase (Hollingsworth and Johns 2003). In case of emulsion, the degree to which that restriction applies is dictated by the size of the droplets, hence molecule in larger droplet could diffuse longer distance than one enclosed in smaller droplet. Hence, higher diffusion coefficient (more freedom for molecule to diffuse) indicates larger droplet size and vice versa (Aichele 2007; Cozzolino et al. 2008; Johns and Gladden 2002). One of the first trail of applying PFG NMR for the sizing of droplets was performed by (Balinov et al. 1994) where they investigated the influence of age, water content, and alcohol content on the emulsion droplet size. The technique proved its effectiveness in terms of monitoring when droplet coalescence dominates compared to droplet flocculation (Peña et al. 2005). NMR technique, in particular, is readily applicable to concentrated emulsions which are opaque or contaminated (dispersed with air bubbles or solid suspension) (Johns 2009). This gives NMR an advantageous property in addition to that it is non- invasive technique as compared to alternative conventional droplet sizing techniques such as optical observation, electrical resistivity, light scattering, and ultrasound spectroscopy measurements.

Sjöblom et al., (2017a) designed by- line benchtop NMR setup to monitor the effectiveness of different commercial demulsifiers in treating water- in- crude oil emulsions. The measurements were performed in three different commercial demulsifiers, good results were reported. More importantly, a rapid version of PFG NMR, Diffrain, was successfully incorporated into the control flow loop where the measured emulsion droplet size was controlled based on the impeller speed of an emulsion mixing device. In addition, the authors argued that these developments could move benchtop NMR closer to direct industrial applications. Various water- in- oil emulsions were tested regards their stability following addition of either sodium chloride (NaCl) or calcium chloride \(\mathrm{CaCl_2}\) (Davis et al. 2021; Doğan et al. 2020; Ling et al. 2018). Different types of oil were used in their study such as paraffin oil, xylene, and modified crude oil in which asphaltene or acidic component was removed, respectively. NMR droplet sizing suggests that the increasing of salt concentration would reduce the growth rate of emulsion droplet size. These measurements were cross validated using interfacial tension measurements that also reduced with increasing of salt concentration.

The ability of injecting \(\mathrm{CO_2}\) into water- in- oil emulsions as a mean of breaking the emulsion was investigated using PFG NMR measurements (Azizi et al. 2019). Essentially, there are two theories governing the phenomena of breaking the emulsion using \(\mathrm{CO_2}\) . The first theory suggests that pressurizing \(\mathrm{CO_2}\) strips the asphaltene of the surface of the droplet, hence, it solubilizes the asphaltene coating from the oil- water interface and hence emulsion destabilization (Zaki et al. 2003). The second theory states that \(\mathrm{CO_2}\) dissolves into the water phase and when \(\mathrm{CO_2}\) is depressurized causing it to come out of solution due to reduced solubility. Then, the \(\mathrm{CO_2}\) would rope the surfaces of the droplet causing coalescence and increase the emulsion droplet size (Sjöblom et al. 2017b). Promising results were also documented for in- situ droplet size study in both high and low salinity water systems after \(\mathrm{CO_2}\) treatment using NMR (Ling et al. 2018). Consequently, NMR has a great potential to serve as valuable production logging tool to diagnose and detect the downhole emulsion.

### Common misinterpretation in conventional rocks

Pore (diffusional) coupling, and the internal magnetic field are two common issues that could mislead the interpretation of the NMR relaxation data for conventional rocks. These issues are due to the heterogeneity and the constitution of the rock pore structure of the conventional reservoir rocks. Four different relaxation rates may need to be taken into consideration include:

The bulk relaxation rate of the fluid in the pores. The relaxation rate in the vicinity of the pore wall. The rate of molecular diffusion to the pore wall while taking into account the bulk diffusion coefficient. The rate for pore coupling. Diffusion due internal field gradients causes additional \(T_{2}\) dephasing induced by the presence of paramagnetic ions or due to pore restriction. Interactions of spins with paramagnetic components on pore walls in smaller pores, this will additionally shortens \(T_{2}\) relaxation time.

However, it is common in most NMR petrophysical analyses models to assume fast diffusion between bulk and surface fluid and neglects the possibility of pore fluid transport to neighboring pores of different dimensions. Thus, for better measurement accuracy, a more generalized relaxation model is required to include the rate of exchange of the fluid between pores (pore coupling) and the rate of diffusion within a pore (Barrie 2000; Grunewald and Knight 2009; Rios et al. 2016; Wang et al. 2018a, b). In most NMR relaxation time studies, it is usually assumed that the rate of molecular diffusion to the pore wall is very large, and a weak pore coupling is often considered for simplicity (Song et al. 2014; Thrane et al. 2019; Yu et al. 2019). It should be noted that the effect of pore coupling on relaxation times has been modeled by several authors (Ghomeshi et al. 2018; McCall et al. 1991; Mitchell et al. 2019).

In this case, the estimation of formation properties such as porosity, permeability and irreducible water saturation using the traditional \(T_{2,cutoff}\) method gives erroneous results. Several studies has investigated the impact of pore geometry and chemistry on diffusion coupling and proposed techniques that can take into account the effects of diffusional coupling for better estimation of properties (Anand and Hirasaki 2005; Grunewald and Knight 2011, 2009; Hinedi et al. 1997). Diffusion due internal field gradients causes additional \(T_{2}\) dephasing which cannot be removed by spin echo, it is a relaxation mechanism that quantify the rate at which hydrogen nuclei in the pore fluid diffuse through local magnetic field induced by the presence of paramagnetic ions or due to pore restriction (Fordham and Mitchell 2018; Hürlimann 1998; Zhang et al. 2016). Interactions of spins with paramagnetic components (which are usually found in clay minerals such as chlorite and illite) on pore walls become significant causes of relaxation in smaller pores, this will additionally influence the transverse relaxation time (it shortens \(T_{2}\) ) (Elsayed et al. 2020a; Livo et al. 2020). The shortening of \(T_{2}\) is a function of paramagnetic species and the pore size (Connolly et al. 2019; Elsayed et al. 2021a; Mitchell et al. 2010). Regarding the impact of clay content on NMR parameters and measurements, several studies concluded that surface relaxivity increases with increase of paramagnetic- rich clay content (D'Agostino et al. 2017; Foley et al. 1996; Keating et al. 2008; Keating and Knight 2007; Li et al. 2019). A reduction in the NMR log- derived porosity as compared to total porosity is argued to be function of chlorite content in sandstone oil reservoir in the North Sea (Rueslatten et al. 1998). They attributed the reduction in porosity estimates to the fine- grained chlorite which induces an internal magnetic field gradient on the pore level causing a significant decrease in the \(T_{2}\) relaxation times. The effect of chlorite was also studied in North Burbank sandstone reservoir and showed strong effect on internal gradient as compared to Berea sandstone (Zhang et al. 1998). Chlorite is specifically important when interpreting NMR data because it contains ferromagnetic and paramagnetic ions and usually exists in form of pore- lining clays. In a recent study (Elsayed et al. 2020a), the impact of clay mineralogy and content on the internal gradient and \(T_{2}\) distribution of sandstones were investigated. The results showed that chlorite has the largest impact on NMR measurements compared to other clay minerals due to the higher paramagnetic content of the former. While no impact was observed for kaolinite content, a strong correlation was found between chlorite content and both internal gradient and shifts in \(T_{2}\) distribution. Nevertheless, the impact of clay distribution was not systemically investigated.

### Unconventional rock characterization using different NMR techniques

Shale rocks are fine- grained sedimentary rocks with high total organic content at least over \(2\%\) and they are featured by low porosity and permeability (Espitalie et al. 1977; Passey et al. 2010; Sondergeld et al. 2010). They are composed of organic and inorganic pore structure. Inorganic pore structures typically comprise of silica, clay, carbonates, and pyrite. There are also verity of organic matters such as kerogen (porous insoluble organics), bitumen (soluble highly viscous organics), low viscosity oils and natural gas (Curtis et al. 2012; Gong et al. 2020; Zhang et al. 2012). Accurate quantification of these organic components are vital in terms of establishing shale reservoirs quality and maturity (Makeen et al. 2021, 2016). NMR studies of shale rocks is a rapidly developing field. For reliable characterization shale rocks using NMR, some relaxation considerations are discussed. Traditional surface relaxivity is expected to dominate in inorganic pore structure; that is small amount of paramagnetic metal species at the pore surfaces interacting with the adsorped layer fluid (Guo et al. 2020; Martinez and Davis 2000; Washburn 2014). The possible NMR relaxation dynamics in organic pore structure is complex and it is still source of disagreement in the literature (Bousige et al. 2016). The relaxation in organic pore structure in shale rocks is affected by (Panattoni et al. 2021a, b; Singer 2013):

- Kerogen maturity; the solid \(^1\mathrm{H}\) density will have a potential effect on the adsorped surface relaxation.
- Kerogen pore size; the nano-sized pores causes additional relaxation coming from molecular confinement effect.
- Diffusive coupling between organic and inorganic pore structures; the relaxation is averged out in case of gas diffusing from inorganic to organic pores instead of showing two distinct pore sizes.
- Origin matter origins; the source of organic matter within the shale rocks. The sources are usually paramagnetic metalloporphyrins such as \(\mathrm{Fe^{+2}}\) , and \(\mathrm{Mn^{2 + }}\) .
- The wetting characteristic of organic and inorganic pore structure; it is usually expected that inorganic pore structure with surface hydrophilic (water-wet), and organic pore structure is expected to be hydrophobic (oil-wet).

In conventional reservoirs, fundamental petrophysical properties such as porosity, permeability, pore size distribution, wettability, and fluid saturation can be estimated accurately. However, in unconventional reservoirs, interpreting NMR signals is more difficult than in conventional formations. It is crucial to understand the characteristics of the solid matrix and fluids in unconventional reservoirs for both resource evaluation and production. Organic content, for instance, comprises hydrogen and hence appears in certain NMR measurements, causing the resulting signal from shales to no be lithology- dependent. Below are some examples from literature discussing the characterization of unconventional rocks using NMR.

[Figure 11: NMR \(\mathrm{T_1 / T_2}\) relaxation map for different fluid type and conditions adapted from (Fleury and Romero- Sarmiento, 2016b)]

Figure 11 illustrates global \(T_{f} / T_{2}\) map in terms of fluid typing of NMR relaxation in shale rocks. This concept directs the interpretation of fluids behavior using NMR in shale studies. No signal is expected to present in the region below the one- to- one because having \(T_{f}\) less than \(T_{2}\) is physically unreasonable in terms of NMR relaxation theory, hence, any signal in this region is interpreted as experimental artifact. In case of free bulk fluids, long relaxation times component are expected, in addition, \(T_{f}\) and \(T_{2}\) relaxation time components are almost identical lying on the \(T_{f} / T_{2} = 1\) line.

[Figure 12: Summary of the different scenarios studied in (Fleury and Romero-Sarmiento, 2016a)]

Fluids in the small pore sizes are expected to appear as we decrease in \(T_{2}\) and a decrease in mobility of fluid is observed as the \(T_{f} / T_{2}\) ratio is increased. (Fleury and Romero- Sarmiento, 2016b) studied the kerogen, clay- bound water, methane, and water \(T_{f} / T_{2}\) signal behavior at \(23\mathrm{MHz}\) for shale rocks. \(T_{f} / T_{2}\) ratio corresponds well with the maturity of the kerogen, hence, lower \(T_{f} / T_{2}\) ratio is expected for immature kerogen compared to oil and gas windows. Relaxation components have been identified for kerogen with \(T_{f} / T_{2}\geq 50\) The clay- bound water which has short \(\mathrm{T}_{2}\) showed \(T_{f} / T_{2}\approx\) 20. Methane within kerogen- isolates pores \((T_{f} / T_{2}\approx 15)\) and confined water \((T_{f} / T_{2} = 2)\) overlap notably in \(T_{f} / T_{2}\) maps as shown in Fig. 12. Furthermore, the kerogen signal contribution was only sensed with high hydrogen content \((>30\mathrm{mg / g})\) which is an important consideration and limitation in the study of kerogen in low- field NMR.

Kausik et al., (2016a) utilized lower NMR- field \((2\mathrm{MHz})\) to study over- mature shale containing no bitumen. At irreducible water saturation, two relaxation populations were observed at \(T_{f} / T_{2} = 2\) ; the short \(T_{f,2}\) times suggest small pore structure assigned to clay- bound water, while longer \(T_{f,2}\) times corresponds to irreducible water in inorganic/ organic pores. The establishment of these two populations allow to determine \(T_{2,cutoff}\) which separate producible and bound fluids in pore space. Re- saturating the same sample with methane at 5000 psi showed again two relaxation populations; small relaxation population at \(T_{2} = 10\) ms and \(T_{f} / T_{2} = 2.6\) interpreted as gas in shale pores, while large relaxation population at long \(T_{f}\) and \(T_{2}\) assigned to bulk- free methane outside the sample. Another native- state shale from Eagle Ford formation containing residual fluids were studied. \(T_{f} / T_{2}\) showed clear separation between fluids with short \(T_{2}\) (probably clay- bound water, bitumen or combination) and fluids with larger \(T_{2}\) such as oil in organic and inorganic pores in this case. However, \(T_{f} / T_{2}\) ratio for oil in organic pores is larger than in inorganic pores. A plausible explanation is that organic pores are expected to be oil- wet leading to larger \(T_{f} / T_{2}\) ratio. A huge change in the signal coming from oil in organic pores was observed after restauration compared to small change in inorganic pores. This indicates that the organic pores within this shale are responsible for the majority of the pore space. In gas and oil shale, kerogen signal was not detected at \(2\mathrm{MHz}\) indicating that kerogen signal requires is not feasible using the current NMR wireline logging technology. However, establishing \(T_{2,cutoff}\) values as shown in red dotted line in Fig. 13 are very useful to help transferability of interpretation of relaxation characteristic back to 1D \(T_{2}\) relaxation distribution, and to increase validity of interpretation of well- logging data.

[Figure 13: Summary of the different scenarios studied in (Kausik et al., 2016b)]

The advantage of performing shale rocks studies at Highfield NMR is the frequency dependence (The magnetic field dependence of \(T_{j}\) compared to \(T_{2}\) . \(T_{j} / T_{2}\) ratios could be totally different for the same shale rock at different magnetic field strength as shown in because \(T_{j}\) is highly dependent on magnetic field strength (Kausik et al. 2017). In case of shale rocks, this provides additional and better discrimination between fluid types that could be difficult to separate in low- field. The results showed that \(T_{j} / T_{2}\) map in high- field NMR (400 MHz) separates very well the kerogen \((T_{j} / T_{2} = 2000)\) and bitumen \((T_{j} / T_{2} = 600)\) signals in Upper Bakken shale. The identities of the peaks in first map in the native sample were confirmed by extracting the two components (kerogen and bitumen) separately, and performing measurement on them individually. Furthermore, this shows that it is much easier to discriminate the identities of these peaks at high- field compared to low- field where they are essentially characterized by approximately same \(T_{j} / T_{2}\) ratio.

A successful attempt was achieved to capture solids and semi- solid NMR which have extremely short \(T_{2}\) (provide reference here please). The relaxation in semi- solids can be partially described as Gaussian rather than exponential, and the mathematics of a very rapidly relaxing system can be expressed as:

\[M_{xy}(t) = \sum_{i}A_{i}\exp \left(\left\{\frac{-t}{T_{2i}^{*}}\right\}^{2}\right) + B_{i}\exp \left(\frac{-t}{T_{2i}}\right) \quad (31)\]

where \(M_{x,y}(t)\) is the acquired signal, \(A_{i}\) is the contribution from Gaussian decay of the \(T_{2}^{*}\) component, and \(B_{i}\) is the contribution from exponential decay of the \(T_{2i}\) component. The magnetization here has two components; Gaussian decay acquired by free induction decay (FID), and the exponential decay acquired by CPMG. This requires new simultaneous Gaussian- Exponential kernel for data inversion described in details in (Washburn et al. 2015). A recent study performed experiments on different shale rocks subjected a range of relative humidity environments in order to differentiate between rapid relaxing signal from organics (bitumen and kerogen) and clay- bound water (Yang et al. 2020). Three peaks were produced as shown in Fig. 14. The first peak at very short relaxation time is attributed to the Gaussian contribution (solid organics), whereas the two other peaks 2, and 3 (water in organic, and inorganic pores, respectively) are attributed to the normal exponential decay. It can be inferred that the Peak 1 area (short \(T_{2}\) components) is essentially independent of the relative humidity of the system which provides validation to the assumption of assigning the peak 1 to solid organics. Peaks 2 and 3 are sensitive to relative humidity where the peak area increase with the increase of relative humidity clearly. In addition, the Gaussian components attributed to solid organics demonstrate strong correlation with organic matter (Figs. 15 16).

[Figure 14: \(T_{2} / T_{2}^{*}\) relaxation distribution as function of relative humidity, adapted from (Yang et al. 2020)]

[Figure 15: Comparison between (a) \(^{13}\mathrm{C}\) MAS spectra, b \(^{1}\mathrm{H}\) MAS spectra, and c High-field \(\mathrm{T}_2\) distribution results as function of Vitrinite reflectance adapted from (Panattoni et al. 2021a, b)]

[Figure 16: LWD-NMR prototype tool. Reprinted with permission from (Prammer et al. 2000a, b)]

A recent study investigated the aromatic to aliphatic ratio of kerogen structures which is an indicator of kerogen maturity, so as kerogen matures, it changes from having aliphatic chains to aromatic components (Panattoni et al. 2021a, b). Different solid- state NMR spectroscopy methods ( \(^{13}\mathrm{C}\) MAS spectra, \(^{1}\mathrm{H}\) MAS spectra, and Highfield \(T_{2}\) distribution) were used by the authors because \(\mathrm{sp}^{3}\) - hybridized and \(\mathrm{sp}^{2}\) - hybridized has different chemical shift resonances (Panattoni et al. 2021a, b). It is worth mentioning that \(\mathrm{sp}^{3}\) - hybridized has diamond- like is indicating aliphatic while \(\mathrm{sp}^{2}\) - hybridized has graphite- like structure indicating aromatic. Error! Reference source not found a) demonstrates \(^{13}\mathrm{C}\) NMR shift as function of Vitrinite reflectance \(R_{o}\) which increases with increasing of shale maturity. Additionally, as the maturity increase, there is a shift from aliphatic to aromatic dominated peaks. Chemical shift resolution in \(^{1}\mathrm{H}\) NMR is much narrower than in \(^{13}\mathrm{C}\) NMR, however, it can still be seen that there is clear shift toward higher chemical shift indicating of the system progressing from aliphatic to aromatic state (Error! Reference source not found b) extremely short bimodal \(T_{2}\) distribution were observed as shown in Error! Reference source not found b) using high field (400 MHz) which is an indicative of solid structure of kerogens. They found out that there is increase in longer \(T_{2}\) population with increasing maturity which is counter- intuitive behavior. However, it was argued that kerogen maturation would result in losing \(^{1}\mathrm{H}\) density meaning, this would enlarge the average \(^{1}\mathrm{H}\) - \(^{1}\mathrm{H}\) dipolar coupling distances resulting in longer \(T_{2}\) . NMR has shown its effectiveness in organic matter quantification (Silletta et al. 2022), has introduced a mixed scheme that combine the tradition CPMG, that senses the long- time decay, with the fast- decaying signal acquisition during the free induction decay (FID). The method was used on several samples obtained from Muerta Formation in the Neuquen Basin, Argentina, with high success. Furthermore, a robust determination of shale maturity can be obtained using combination of High- field and \(^{13}\mathrm{C}\) NMR spectroscopy (Song and Kausik 2019).

## NMR applications in field scale

NMR applications in field scaleNMR technology has been widely used for the oil industry for borehole logging. At the early stage of utilizing this technology for the industry, the earth filed was adapted (Kleinberg and Jackson 2001). Those tools were highly influenced by the drilling fluids in the borehole, which reduced the reliability of the tool. Jasper Jackson introduced an alternative way called, inside- out, of performing NMR in 1980 (Jackson et al. 1980). Although, the NMR has many technical difficulties and challenges of performing measurements downhole, it can be obtained during drilling. The advantage of NMR while drilling is the ability to provide information of the uninvaded zones of the reservoirs (Sun et al. 2020). The first version of the LWD- NMR was introduced in 1990, since then, there have been several improvements to the tool. The recent tools include, proVISION and proVISION Plus (Schlumberger) (Horkowitz et al. 2002), MRILWD (Halliburton) (Prammer et al. 2002), Mag- Trak (Baker Hughes) (Borghi et al. 2005), small borehole LWD- NMR tools (Baker Hughes and Saudi Aramco) (Akkurt et al. 2009; Kruspe et al. 2009) and large- diameter LWD- NMR tools (Baker Hughes and Schlumberger) (Bachman et al. 2016; Coman et al. 2014). Each tool has different arrays of paramount magnets and different functions.

These tools are classified into two categories based on the structure of the magnet (Sun et al. 2020). First type that has a longer length and can produce a relatively high gradient of dipole stray magnetic field about \(14\mathrm{G / cm}\) . Hence, the tool can only measure \(T_{2}\) when it is in the sliding state. However, in case of high radial vibration during drilling, \(T_{1}\) can only be measured in this case. This type of magnet has a long sensing area along the well axis that provides the condition for the tool to take measurements while lifting with suitable speed.

The second type of magnet structure is based on the "Jasper Jackson". This magnet produces a natural axisymmetric annular low gradient field (Demas et al. 2008). In case of strong radial vibration this tool is more suitable for \(T_{2}\) measurements. The new enhancement on this tool magnet design is mainly focused on increasing the low gradient area width in the annular magnetic field. This is to achieve the \(T_{2}\) measurement's purpose even if the tool is in the drilling state with a strong radial vibration (Coman et al. 2018). In general, the LWD- NMR magnets are design to reduce the effect of the radial vibration on \(T_{2}\) , achieving one- directional measurement and increasing the logging speed. The key advantage of the LWD- NMR is the ability to acquire the advanced information evaluation such as continues mobility information of the formation in real time during the drilling operation (Selheim et al. 2017).

### Design and improvements in LWD-NMR

The first tool of LWD- NMR prototype was designed and field tested in 1999 by (Prammer et al. 2000a, b). The tool was successfully run for \(130\mathrm{h}\) downhole, which included interval of harsh drilling conditions. The data obtained from the tool showed that the LWD- NMR provide a source- free porosity, permeability and free fluid data compared to MRIL wireline logs. As shown in Error! Reference source not found. The tool was designed for \(8 - 1 / 2\) to \(10 - 5 / 8\) in bit size with two main sections and total length of \(42\mathrm{ft}\) . The sensors are in the lower portion of the tool while the electronics are in the upper section.

A high strength collar made of steel interfaces with the drill string though a standard \(4 - 1 / 2\) IF connections (Internal- flush style connection). The collar contains the magnet and the flow tube which is in the sensor section as well as the radio- frequency (RF) antenna outside the collar that is 24 inch long. Furthermore, the antenna is embedded with fiber glass and rubber and a Wear pads are located above and below the antenna to stand off the rubber section.

The upper section that contains the electronic, is comprised of power conditioning, processing, control, memory, batteries and RF transmitter and receiver. Moreover, the tool has a two- sidewall readout (SWRO) ports for the application of external power and accessing the tool' data memories and program. The tool was tested for two fields data obtained from GRI Catoosa Test Facility (Prammer et al. 2000a, b), a new hole was drilled and logged with wireline. Then it was re- drilled and logged by the designed LWD- NMR tool. The tests showed that the response of the developed tool resembles the wireline values in sand, shale and fracture sequence as shown in Fig. 17. This indicates the ability of obtaining NMR measurement while drilling. Furthermore, the second test tool can replicate the complete MRIL logs after drilling, showing the robustness of the measurement against rotation and vertical or lateral motion.

[Figure 17: Log response results of the LWD test. Reprinted with permission from (Prammer et al. 2000a, b)]

A new improvement in the tool has been introduced by Heidler et al. (2003), which operates in sliding, station and drilling state. The tool (proVISION) provides a real- time information of the wellbore trajectory optimization. According to (Coman et al. 2014), the early prototype introduced by (Prammer et al. 2000a, b) is high magnetic field gradient (HG) tool. Hence, the tool is primarily used for \(T_{1}\) measurements. The gradient field is produced by the opposing dipoles with azimuthal symmetry. A sensitive region with thin shell of height 6 inch and diameter of 14 inch is produced through the interaction of the static magnetic field and the radio- frequency field. According to (Heidler et al. 2003), the depth of investigation depend on the tool centralization and borehole size. After tests, it was found that the tool provides \(T_{2}\) measurement with minimal impact on the drilling process. In addition, harsh drilling conditions are detected by the motion sensors which provides a log quality control for NMR. Finally, these sensors provide information on tool motion either sliding, rotating or stationary.

Another tool developed by Coman et al. (2014) for large boreholes provides a better vertical resolution and improvement of the \(T_{2}\) measurements during the lateral motion by short inter- echo time of \(0.4\mathrm{ms}\) . The design of the tool for large borehole elevated the operating frequency as well as increased the signal to noise ratio for improved vertical resolution. The presented 8.25 inch tool is LG, powered by alternator. The diameter of the tool and the depth of investigation are fixed, and the sensor outer diameter (SOD) is constrained by several requirements such as mechanical robustness, minimal borehole size and minimal required flow, and the tool technical data is presented in Table 5.

Table 5 Tool technical data (Coman et al. 2014)

| Property | Value |
|----------|-------|
| Tool size | 8.25 in |
| Approximate MR frequency | 460 kHz |
| Outer diameter of the sensor section | 10.625 in |
| Maximum temperature | 302°F |
| Minimum mud resistivity | 0.1 Ω-m |
| Maximum pressure | 30,000 psi |
| Diameter of investigation | 18 in |
| Aperture | 4.6 in |
| Inter-echo time for echo-trains | 0.6 ms |
| Inner diameter of investigation | 16.5 in |
| Effective magnetic gradient | 2.4 G/cm |

According to the authors (Coman et al. 2014), increasing SOD, reduces the thermal noise, and the power requirements for the radio- frequency and increases the MR signal. This leads to shorter in the ringdown time and increased in SNR because of the induction law, the tool shows a good agreement between the LWD tools and the proposed tool design.

Bachman et al. (2016) tested a proposed LWD- NMR tool for large borehole based on the principle of the tool introduced by Heaton et al. (2012) for low mud flow rate. The tests were carried for a well of 12.25- inch drilled with WBM in Ecuador and 12.25- in borehole drilled with OBM for deep water along the Gulf of Mexico. The tool was able to identify the hydrocarbon zones that were left undetected and added as considerable amount to the reserve estimation. Furthermore, by comparing the LWD porosity and permeability, the match was excellent with the core analysis. According to the results, it was found that LWD- NMR was able to aid in optimizing the wireline logging program, in addition, the tool was run successfully in deviated well sections.

The LWD- NMR porosity data usually is not affected by the lateral motion of the tool, however, according to Coman et al. (2018) some other parameters such as the bound water, movable fluid, viscosity and permeability might still be affected. Thus, they introduced, a data- based lateral motion correction (LMC) to account and quantify the effect of the lateral motion, hence improving the porosity estimation derived from \(T_{2}\) distribution.

Despite the improvements on the NMR logging while drilling tools, still the measurement is highly affected by the complex motion during drilling like radial vibration, axial rotation, and longitudinal drilling. However, radial vibration has the greatest effect on the \(T_{2}\) measurement (Morley et al. 2002). According to Sun et al. (2020), some methods to reduce the effect of motion on the measurements exist. These methods include optimization of mechanical structure, optimization of sensor design and optimization of detection mode. They include shortening the distance between the borehole wall and the tool by adopting the mechanical centering stabilizer, thereby restricting the range of the radial vibration. The optimization of detection mode method aims to minimize the effect of the complex magnetic fields and the displacement of sensing area during the acquisition of the adjacent echoes by shortening the echo time and reduce the number of echoes. The method of optimizing the sensor design aims to reduce the impact of the magnetic field variation of the spins while maintaining the same radial displacement by reducing the gradient of the magnetic field. Moreover, it increases the width of the sensing area by increasing the RF bandwidth.

As an enhancement to the tool design, Sun et al. (2020) focused on the optimization of the sensor design method for better tolerance for vibration and high reliability while utilizing the " Jackson Jasper" magnet structure shown in Fig. 18. The tool structure consists of two polarized apposed cylindrical polarized magnets arranged along the axial direction with a space between them. A ring saddles around the axial is generated at the center of the gap with a zero magnetic field gradient at the center and the radial reaches the maximum. The gradient inside the saddle is low and form a large resonant area for NMR measurement.

[Figure 18: "Jackson Jasper "magnet structure and magnet field distribution. Reprinted with permission from (Sun et al. 2020)]

In LWD- NMR tool, the field distribution should have sufficient field strength of about 114.4G, suitable detection depth of about \(16\mathrm{cm}\) and sufficient width of the low gradient area about \(50~\mathrm{mm}\) . Thus, (Sun et al. 2020) optimized the magnet design under these conditions by changing the geometric parameters to generate the required magnetic field for better LWD- NMR measurements. The introduced magnet structure LWD- NMR tool proved to poses high vibration tolerance and reliability. The tool has DHCM (double hollow cylinder magnet) (Sun et al. 2018) structure and solenoid- optimized RF antenna. The authors found that by reducing the gap distance and increasing the length of the magnets at the ends, it could increase the strength of the magnetic field. However, the growth of the magnetic field is reduced after the length exceeds \(350~\mathrm{mm}\) while the strength decreases sharply with the increasing the gap as shown in Fig. 19.

[Figure 19: Effect of length on the magnetic field Reprinted with permission from (Sun et al. 2020)]

Further improvements on the data acquired by the LWD- NMR was done by Jachmann et al. (2020), to account for the drilling environment as it poses challenges for real- time data processing and porosity calculations. They introduced a new split inversion methodology for such purpose. The method has several steps: stacking the data, first inversion, data reconstruction, compression, transmission, decompression, and second inversion. Stacking the data first is to increase the signal to noise ratio (SNR) before the inversion. The lab results showed the accuracy of this method in calculating the porosity while accounting for the motion of the tool, as it was expected increasing the ROP increased the porosity losses. The split inversion was able to correct the porosity estimation.

### Field implementation and applications

#### Tar detection in carbonates

During the drilling, Tar must be detected as early as possible, which requires LWD technologies. Tar is characterized by a specific gravity of less than \(10^{\circ}\) and viscosity of more than \(10,000\mathrm{cp}\) . The identification of tar is important for reserve estimation as it is classified as an unrecoverable resource. Additionally, tar can be damaging for reservoir performance because it acts as a permeability barrier in addition to reducing the total porosity. The lateral distribution of the tar may not be continuous as a tar mat in many cases. In addition, the tar top may sometimes be encountered at different depths. These obstacles make it a high priority to detect tar at the early stage of drilling to minimize the change in the drilling program.

According to Tester et al. (2009), the conventional triple combo LWD is not very useful by itself, since it cannot differentiate tar based on water saturation and/or porosity neither the mobility information can make this detection as well. Furthermore, the use of the geological information extracted from the cutting may also have limited application in real- time tar detection because of the uncertainties in the depth determination in some of the wells.

Tester et al. (2009) proposed a 4 steps work follow for real- time tar detection based on the integration of the triple combo, NMR, and formation tester along with the geological information from the cuttings for carbonate reservoir in the Middle East. The process involved the calculation of two indicators the missing porosity and excess- bound fluid for flagging potential tar- baring zones. Then it is followed by the mobility data from the formation tester. The process started by calculating the neutron and the density porosity, then the water saturation is calculated using Archie's and the total porosity. After that the missing porosity is calculated by taking the difference between the total and the NMR porosity. Based on that, if the tar exists in a zone, the NMR signal will decay too fast due the almost solid state of tar resulting in porosity that is less than the actual. This method was tested in one of the wells that penetrated several carbonate reservoirs. The tool included density, neutron, resistivity imaging, NMR tools and formation tester.

NMR- LWD allows the petrophysical information to be obtained during drilling operation before any invasion happens. This reduces the borehole risk while allowing for timely decision (Seifert et al. 2007). In 2007 Saudi Aramco ran LWD- NMR, conventional NMR and NMR fluid analyzer in one of the fields, the NMR fluid analyzer was run to evaluate the LWD- NMR results. The filed posed some challenges as two of the three main reservoirs contained an extensive tar mat near the oil- water contact. The LWD- NMR was compared against wireline NMR, density, and neutron porosity for lower, middle, and upper section of the well, the porosity estimation from the LWD is in good agreement with the neutron and density porosity.

#### Geosteering and well placement optimization

Geosteering horizontal section in thin sand was achieved by (Oguntona et al. 2004), using LWD- NMR and resistivity. Both LWD tools were used to geosteer the horizontal section and geo- stop the horizontal well after drilling an optimal

[Figure 20: Pilot hole logging data. Reprinted with permission from (Oguntona et al. 2004)]

rain hole length through a thin sand in Niger Delta. Based on the NMR results an optimized horizontal sidetrack was drilled in the high permeable section of the sweet spot of the reservoir. The first case was Meji field which is an offshore well located in Nigeria. The objective of this case was to drill 1500 ft lateral section through the most productive part of the sand. As shown in Fig. 20, the NMR measurements for the drilled pilot hole confirms that the sand is clean and large pore size.

The second case was to target thin bed sand which is 5 ft thick, with horizontal well of total length 1200 ft. the plan was to land the well within 7.2 ft thin reservoir. Then a lateral section of 1200 ft is drilled after he is casing with \(90^{\circ}\pm 2^{\circ}\) inclination within the sand. The data from the NMR of the pilot hole confirmed that the sand thickness is permeable and good enough targeted production. The test was conducted successfully with total of 15 adjustments of the 1400 ft lateral well being drilled which increased the well production up by \(33\%\) (Oquntona et al. 2004).

In 2012, LWD- NMR was tested in one of the Brazilian horizontal fields as geosteering tool along with neutron, density, gamma ray LWD (Ribeiro et al. 2012). The filed is \(1000\mathrm{m}\) and targeted net pay of \(65\%\) minimum. The distance between the landing of the horizontal well and the discovery well was \(2\mathrm{km}\) . An S shape pilot well was drilled to reduce the geological uncertainties and establish potential layers. NMR data was used for net pay count and pore- typing. Spectroscopy was used for elemental analysis and images for dips computation and inclination adjustment.

In another case where LWD- NMR was used in a comprehensive logging program for geosteering horizontal well in a complex carbonate reservoir (Kanfar 2012), the reservoir had a low permeability zone in the middle rimmed by tar mat. This makes the objective of optimally placing horizontal producer and injector. The comprehensive program incorporated the data from Formation Pressure While drilling (FPWD) for real- time detection of low mobility zone penetration which will help in adjusting the trajectory of the well. Also, the program utilizes the data from LWD- NMR for heavy oil detection as it gives a continuous information of the fluid mobility of the formation being drilled (Selheim et al. 2017) where the placement of water injector is unlikely. The main purpose of the LWD- NMR was identifying the tar mat zones and quantify the fluid viscosity (Akkurt et al. 2010).

Historically LWD- NMR most common application is heavy oil characterization in carbonate reservoirs (Hursan et al. 2015; Lyngra et al. 2015). A work reported by (Hursan et al. 2016), where the LWD- NMR was used for rock quality and evaluation of clastic reservoir. LWD- NMR was used to maximize the reservoir contact. NMR showed an excellent sensitivity to pore size distribution even with clean sand for optimum geosteering. Furthermore, the LWD- NMR demonstrated its importance in optimizing the well placement and maximize the reservoir contact.

Another field application to identify the bypassed reservoir in limestone for better carbonate sequence evaluation, Petroamazonas EP company utilized LWD- NMR (Pro VISION Plus 8) tool in one of its assets in Ecuador (Morales et al. 2016). The section of the well that was evaluated was behind a \(95 / 8^{\circ}\) casing and included, U, T sands, Basal Tena, and limestone A. The goal of using LWD- NMR was to explore potential permeability of limestone lithology and permeability without the need to use radioactive sources besides the information provided on the pore geometry and fluid distribution. The tool was run based on CMPG pulse sequence with defined echo spacing cycle and wait time. LWD- NMR readings proved to be efficient in discovering the secondary objectives and add to the productivity of the well.

Additionally, another field study for reservoir characterization and Geosteering was conducted in one of the AbuDhabi offshore \(3^{\mathrm{rd}}\) order regressive carbonate sequence wells by (Serry et al. 2016). The reservoir shows wide range of facies. The objective was to drill a highly deviated water injection drain across the sublayers of the reservoir. Moreover, the objective of the geosteering was to place the near horizontal wellbore within a predefined number of geological layers. The deployment of LWD- NMR allowed a rig time optimization and enhanced data quality, since the data is recorded before excessive mud filtrate invasion occurs. Finally, the NMR provided petrophysics- driven geosteering operation along reservoir's highest quality units (Serry et al. 2016).

## Future research directions

Extensive efforts have been done across the academia and industry in the application of NMR with the most significant and important development being given especially to petrophysical and well- logging applications. However, most of these studies are more directed toward applications. A promising area of research is the study of NMR behavior using numerical simulation in the rock core to determine 1D and 2D NMR data. This would allow us to examine several scenarios that could be found in the formation. Furthermore, the development of robust numerical simulation models would save time for the lengthy experiments (such as diffusion using 2D NMR) in the 2 MHz system. Few research groups devolved numerical simulation of NMR response to predict \(T_{2}\) and \(\rho_{2}\) where a reasonable overlap with experimental data was achieved. However, for accurate predictions, all relaxation phenomena should be included without ignoring the effect of bulk and internal gradient effect. This is a topic we are looking into currently, and first findings are promising.

Furthermore, magnetic resonance imaging (MRI) is capable of spatially resolving the distribution of fluid phases (water, oil, and gas). As a result, before piloting operations in a reservoir, laboratory- scale MRI is utilized to assess the performance of different oil recovery strategies, such as chemicals (acids, polymers), sc- CO2, or miscible gas (Lai et al. 2020; Li et al. 2017; Y. Zhao et al. 2020a, b). Studies of miscible fluid injection, such as monitoring \(^{23}\mathrm{Na}\) to analyze oil recovery with varying salinity brine, are made possible by the capacity to probe nuclei other than \(^{1}\mathrm{H}\) (Yang and Kausik 2016). Wettability maps, another key tool for core analysis, is enabled by current research into spatially resolved relaxation and diffusion, which leads to a better knowledge of oil recovery processes. Recent developments in MRI techniques will allow the measurement of rock structure (pore and grain size distributions) and fluid flow (flow propagators) on the same core- plug (Karlsons et al. 2021). These lab- scale data are crucial for enhancing models that predict reservoir- scale effectiveness of oil recovery technologies, and the results are expected to yield future insights into the structure- transport interactions that drive fluid flow in reservoirs.

In addition, an important area of research is the optimization and modifications of pulse sequences that would greatly help in decreasing the data acquiring time which will save millions of dollars for the operational cost. This involves enhancing the signal to noise ratio which is a key parameter to obtain accurate measurements. Finally, in order to determine the application of laboratory findings, the development of a more efficient LWD hardware to mimic the lab results must be investigated.

## Concluding remarks

This paper reviews wide range of NMR applications in the oil and gas industry including laboratory and field measurements. The following bullet points draw the conclusion of this review based on the literature and the previous author's studies in NMR:

- NMR offers powerful tool to understand the rock, fluid, and rock-fluid interactions behavior in porous media. Due to this, it is considered as research trend in the oil and gas industry. 1D and 2D NMR measurements could be implemented in laboratory and in-situ which in turn, provides reliable cross-validation technique between rock core and logs data. Detailed explanation of the pulse sequences, experimental parameter, and acquired signal inversion were discussed in detail.
- NMR applications in petroleum industries especially in the petrophysical characterization of reservoir properties is promising. NMR serves as a strong means of petrophysical characterization of reservoir fluids and rocks in the laboratory core analysis as well as in field logging tools; both in conventional and unconventional plays. This involves porosity, pore size distribution, permeability, fluid saturation capillary pressure, and wettability. NMR measurements has been applied to understand and explore the nature of petroleum reservoir formation, its fluid content, and the most effective and efficient way of production Promising results with good agreement when compared with other conventional methods were being reported as summarized in this review.
- NMR technique plays a significant role in understanding the interactions between the injected fluids and reservoir rocks during EOR operations. Most of the NMR measurements are conducted at ambient conditions using small core samples. Applying NMR measurements using long core samples and at high-pressure high-temperature conditions would be more representative of the reservoir conditions. Also, more NMR research are needed for a better understanding of the rock- fluid interactions during polymer flooding.
- Several interesting topics related to drilling, production engineering and unconventional characterization were discussed in details. In addition, two of the common issues (pore coupling, and internal magnetic field gradient) related to conventional rocks interpretations were reviewed showing their effect in the interpretation of \(T_{2}\) results. For the drilling engineering application, NMR technology was utilized to assess formation damage induced by drilling fluid invasion in addition to mitigation or removal techniques. In oilfield emulsions, the essence of using PFG NMR to measure the emulsion droplet size is by measuring the restricted diffusion caused by the continuous phase. Different NMR techniques such as 1D NMR \((T_{2}^{*},T_{2},^{13}\mathrm{C}\) chemical shift, \(^{1}\mathrm{H}\) chemical shift, and 2D NMR \((T_{f} / T_{2})\) are used to understand the different chemistries existing in shale formations.
- Logging while drilling nuclear magnetic resonance LWD- NMR, showed a great a potential in many areas. Since the development of the tool in 1999 and there have been several improvements on the tool design or the data acquisition methods. The tool provides an insightful real- time information of the formation being drilled as well as the fluid viscosity. This has led a great utilizing the tool in characterizing the lithology of the formation and the type of the pore systems exits. Furthermore, the tool demonstrated its importance as a geosteering tool. Since, the tool can be used for tar mat detection and low permeability, hence, the tool helps in optimizing the well placement and maximizing the reservoir contact.

Funding The authors received no financial support for the research. The Journal of Petroleum Exploration and Production Technology hascovered the publication fees for this article.

## Declaration

Conflict of interest The authors confirm that they do not have any conflict of interest regarding this work.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/licenses/by/4.0/.

## References

[1] Abdul Jameel AG (2021). Identification and quantification of hydrocarbon functional groups in gasoline using 1H-NMR spectroscopy for property prediction. Molecules 26:6989. <https://doi.org/10.3390/molecules26226989>

[2] Adebayo AR, Bageri BS (2020). A simple NMR methodology for evaluating filter cake properties and drilling fluid-induced formation damage. J Pet Explor Prod Technol 10:1643–1655. <https://doi.org/10.1007/s13202-019-00786-3>

[3] Adebayo AR, Bageri BS, Al Jaberi J, Salin RB (2020a). A calibration method for estimating mudcake thickness and porosity using NMR data. J Pet Sci Eng 195:107582. <https://doi.org/10.1016/j.petrol.2020.107582>

[4] Adebayo AR, Isah A, Mahmoud M, Al-Shehri D (2020). Effects of foam microbubbles on electrical resistivity and capillary pressure of partially saturated porous media. Molecules 25(15):3385. <https://doi.org/10.3390/molecules25153385>

[5] Afsahi B, Kantzas A (2007). Advances in diffusivity measurement of solvents in oil sands. J Can Pet Technol. <https://doi.org/10.2118/07-11-05>

[6] Aichele CP, Flaum M, Jiang T, Hirasaki GJ, Chapman WG (2007). Water in oil emulsion droplet size characterization using a pulsed field gradient with diffusion editing (PFG-DE) NMR technique. J Colloid Interface Sci 315:607–619. <https://doi.org/10.1016/j.jcis.2007.07.057>

[7] Akkurt R, Marsala AF, Seifert D, Al-Harbi A, Buenrostro C, Kruspe T, Thern HF, Kurz G, Blanz M, Kroken A (2009).Collaborative development of a slim LWD NMR tool: from concept to field testing. In: All Days. SPE. <https://doi.org/10.2118/126041-MS>

[8] Akkurt R, Seifert D, Eyvazzadeh R, Al-Beaiji T (2010). From molecular weight and NMR relaxation to viscosity: an innovative approach for heavy oil viscosity estimation for real-time applications. Petrophys SPWLA J Form Eval Reserv Descr 51.

[9] Al Harbi AM, Gao J, Kwak HT, Abdel-Fattah AI (2017). The study of nanosurfactant EOR in carbonates by advanced NMR technique. In: Day 4 Thu, November 16, 2017. SPE. <https://doi.org/10.2118/188710-MS>

[10] Al-Garadi K, El-Husseiny A, Elsayed M, Connolly P, Mahmoud M, Johns M, Adebayo A (2022). A rock core wettability index using NMR T2 measurements. J Pet Sci Eng 208:1093. <https://doi.org/10.1016/j.petrol.2021.109386>

[11] Allsopp K, Wright I, Lastockin D, Mirotchnik K, Kantzas A (2001). Determination of oil and water compositions of oil/water emulsions using low field NMR relaxometry. J Can Pet Technol. <https://doi.org/10.2118/01-07-05>

[12] Al-Mahrooqi SH, Grattoni CA, Moss AK, Jing XD (2003). An investigation of the effect of wettability on NMR characteristics of sandstone rock and fluid systems. J Pet Sci Eng 39:389–398. <https://doi.org/10.1016/S0920-4105(03)00077-9>

[13] Al-Mahrooqi SH, Grattoni CA, Muggeridge AH, Zimmerman RW, Jing XD (2006). Pore-scale modelling of NMR relaxation for the characterization of wettability. J Pet Sci Eng 52:172–186. <https://doi.org/10.1016/j.petrol.2006.03.008>

[14] Almenningen S, Roy S, Hussain A, Seland JG, Ersland G (2020). Effect of mineral composition on transverse relaxation time distributions and MR imaging of tight rocks from offshore Ireland. Minerals 10:232. <https://doi.org/10.3390/min10030232>

[15] Al-muntasheri GA (2008). Polymer gels for water control: NMR and CT scan studies.

[16] Al-Muthana AS, Hursan GG, Ma SM, Valori A, Nicot B, Singer PM (2012). Wettability as a function of pore size by Nmr. Soc Core Anal 1–12.

[17] Anand V, Hirasaki GJ (2005). Diffusional coupling between micro and macroporosity for NMR relaxation in sandstones and carbonates. In: SPWLA 46th annual logging symposium.

[18] Arnold J, Clauser C, Pechnig R, Anferova CS, Anferov V, Blümich B (2006). Porosity and permeability from mobile NMR core-scanning. Petrophys - SPWLA J Form Eval Reserv Descr 47.

[19] Arnold J, Clauser C, Blümich B (2007). Mobile NMR for rock porosity and permeability. Fak. für Georessourcen und Mater. Doctor, 94.

[20] Arns CH, Sheppard AP, Saadatfar M, Knackstedt MA (2006). Prediction of permeability from NMR response: surface relaxivity heterogeneity. In: SPWLA 47th annual logging symposium.

[21] Arora S, Horstmann D, Cherukupalli P, Edwards J, Ramamoorthy R, McDonald T, Bradley D, Ayan C, Zaggas J, Cig K (2010). Single-well in-situ measurement of residual oil saturation after an EOR chemical flood. In: All Days. SPE. <https://doi.org/10.2118/129069-MS>

[22] Azizi A, Aman ZM, May EF, Haber A, Ling NNA, Husin H, Johns ML (2019). Emulsion breakage mechanism using pressurized carbon dioxide. Energy Fuels 33:4939–4945. <https://doi.org/10.1021/acs.energyfuels.9b00606>

[23] Azizoglu Z, Posenato Garcia A, Newgord C, Heidari Z (2020). Simultaneous assessment of wettability and water saturation through integration of 2D NMR and electrical resistivity measurements. In: Day 4 Thu, October 29, 2020. SPE. <https://doi.org/10.2118/201519-MS>

[24] Bachman N, Jain V, Gunawan Y, Bonningue P, Hazboun N, Kuptsov K, Terán N, Bastidas M, Morales O, Sánchez FM, David R, Montoya G (2016). A new large hole nuclear magnetic resonance logging while drilling tool for early stage formation evaluation. In: 2016 SPWLA 57th annual logging symposium.

[25] Bageri BS, Adebayo AR, Barri A, Al Jaberi J, Patil S, Hussaini SR, Babu RS (2019). Evaluation of secondary formation damage caused by the interaction of chelated barite with formation rocks during filter cake removal. J Pet Sci Eng 183:106395. <https://doi.org/10.1016/j.petrol.2019.106395>

[26] Bageri BS, Adebayo AR, Al Jaberi J, Patil S, Salin RB (2021). Evaluating drilling fluid infiltration in porous media – comparing NMR, gravimetric, and X-ray CT scan methods. J Pet Sci Eng 198:108242. <https://doi.org/10.1016/j.petrol.2020.108242>

[27] Baldwin B, Spinler E (1998). A direct method for simultaneously determining positive and negative capillary pressure curves in reservoir rock. J Pet Sci Eng 20:161–165. <https://doi.org/10.1016/S0920-4105(98)00016-3>

[28] Baldwin BA, Yamanashi WS (1989). Detecting fluid movement and isolation in reservoir core with medical NMR imaging techniques. SPE Reserv Eng 4:207–212. <https://doi.org/10.2118/14884-PA>

[29] Balinov B, Urdahl O, Söderman O, Sjöblom J (1994). Characterization of water-in-crude oil emulsions by the NMR self-diffusion technique. Colloids Surfaces A Physicochem Eng Asp 82:173–181. <https://doi.org/10.1016/0927-7757(93)02618-O>

[30] Banavar JR, Schwartz LM (1987). Magnetic resonance as a probe of permeability in porous media. Phys Rev Lett 58:1411–1414. <https://doi.org/10.1103/PhysRevLett.58.1411>

[31] Barrie PJ (2000). Characterization of porous media using NMR methods. In: Annual reports on NMR spectroscopy. Academic Press Inc., pp. 265–316. <https://doi.org/10.1016/S0066-4103(00)41011-2>

[32] Benavides F, Leiderman R, Souza A, Carneiro G, de Vasconcellos B, Azeredo R (2020). Pore size distribution from NMR and image based methods: a comparative study. J Pet Sci Eng 184:106321. <https://doi.org/10.1016/j.petrol.2019.106321>

[33] Bloembergen N, Purcell EM, Pound RV (1948). Relaxation effects in nuclear magnetic resonance absorption. Phys Rev 73:679–712. <https://doi.org/10.1103/PhysRev.73.679>

[34] Blümich B (2019). Low-field and benchtop NMR. J Magn Reson 306:27–35. <https://doi.org/10.1016/j.jmr.2019.07.030>

[35] Blümich B, Anferova S, Pechnig R, Pape H, Arnold J, Clauser C (2004). Mobile NMR for porosity analysis of drill core sections. J Geophys Eng 1:177–180. <https://doi.org/10.1088/1742-2132/1/3/001>

[36] Borghi M, Porrera F, Lyne A, Kruspe T, Krueger V (2005). Magnetic resonance logging while drilling streamlines reservoir evaluation. In: SPWLA 46th annual logging symposium.

[37] Borgia GC, Brancolini A, Camanzi A, Maddinelli G (1994). Capillary water determination in core plugs: a combined study based on imaging techniques and relaxation analysis. Magn Reson Imaging 12:221–224. <https://doi.org/10.1016/0730-725X(94)91522-9>

[38] Bousige C, Ghimbeu CM, Vix-Guterl C, Pomerantz AE, Suleimenova A, Vaughan G, Garbarino G, Feygenson M, Wildgruber C, Ulm F-J, Pellenq RJ-M, Coasne B (2016). Realistic molecular model of kerogen's nanostructure. Nat Mater 15:576–582. <https://doi.org/10.1038/nmat4541>

[39] Bowers MC, Ehrlich R, Howard JJ, Kenyon WE (1995). Determination of porosity types from NMR data and their relationship to porosity types derived from thin section. J Pet Sci Eng 13:1–14. <https://doi.org/10.1016/0920-4105(94)00056-A>

[40] Brautaset A, Ersland G, Graue A, Stevens J, Howard J (2008). Using MRI to study in situ oil recovery during CO2 injection in carbonates. In: Int. Symp. Soc. Core Anal. Abu Dhabi, UAE, 29 Oct. - 2 Novemb. 2008 SCA paper 2008–41.

[41] Broche LM, Ross PJ, Davies GR, MacLeod M-J, Lurie DJ (2019). A whole-body fast field-cycling scanner for clinical molecular imaging studies. Sci Rep 9:10402. <https://doi.org/10.1038/s41598-019-46648-0>

[42] Brown RJS, Fatt I (1956). Measurements of fractional wettability of oil fields' rocks by the nuclear magnetic relaxation method. In: fall meeting of the petroleum branch of AIME. Society of Petroleum Engineers, Los Angeles, California, p. 4. <https://doi.org/10.2118/743-G>

[43] Brownstein KR, Tarr CE (1979). Importance of classical diffusion in NMR studies of water in biological cells. Phys Rev A 19:2446–2453. <https://doi.org/10.1103/PhysRevA.19.2446>

[44] Bryan JL, Mai AT, Hum FM, Kantzas A (2006b). Oil and water content measurements in bitumen ore and froth samples using low field NMR. SPE Reserv Eval Eng 9:654–663. <https://doi.org/10.2118/97802-PA>

[45] Bryan J, Kantzas A, Badry R, Emmerson J, Hancsicsak T (2006a). In-situ viscosity of heavy oil: core and log calibrations. In: Canadian international petroleum conference. Petroleum Society of Canada.

[46] Cai Y, Liu D, Pan Z, Yao Y, Li J, Qiu Y (2013). Petrophysical characterization of Chinese coal cores with heat treatment by nuclear magnetic resonance. Fuel 108:292–302. <https://doi.org/10.1016/j.fuel.2013.02.031>

[47] Callaghan PT (2011). Translational Dynamics and Magnetic Resonance. Oxford University Press, Oxford.

[48] Callaghan PT (1993). Principles of nuclear magnetic resonance microscopy. Oxford University Press, Oxford.

[49] Carr HY, Purcell EM (1954). Effects of diffusion on free precession in nuclear magnetic resonance experiments. Phys Rev 94:630–638. <https://doi.org/10.1103/PhysRev.94.630>

[50] Chen Q, Balcom BJ (2005). Measurement of rock-core capillary pressure curves using a single-speed centrifuge and one-dimensional magnetic-resonance imaging. J Chem Phys 122:214720. <https://doi.org/10.1063/1.1924547>

[51] Chen J, Hirasaki GJ, Flaum M (2006). NMR wettability indices: effect of OBM on wettability and NMR responses. J Pet Sci Eng 52:161–171. <https://doi.org/10.1016/j.petrol.2006.03.007>

[52] Chen P, Wang L, Zhang S, Fan J, Lu S (2018a). Experimental investigation on CO2 injection in block M. J Chem 2018:1–7. <https://doi.org/10.1155/2018/8623020>

[53] Chen Q, Balcom BJ (2008). (12) United States Patent 2.

[54] Chen S, Beard D, Gillen M, Fang S, Zhang G (2003). MR explorer log acquisition methods: petrophysical-objective-oriented approaches. In: SPWLA 44th annual logging symposium.

[55] Chen S, Li L, Shao W, Reiderman A, Balliet R (2018b). Systematic optimization approach for high-resolution NMR logging. In: SPWLA 59th annual logging symposium.

[56] Cheng Y, Chen S, Eid M, Hursan G, Ma S (2017). Determination of permeability from NMR T1/T2 ratio in carbonates. In: SPWLA 58th annual logging symposium.

[57] Coates GR, Marschall D, Mardon D, Num R (1997). A new characterization of bulk-volume irreducible using magnetic resonance. Log Anal. 39(01).

[58] Coates GR, Peveraro RCA, Hardwick A, Roberts D (1991). The magnetic resonance imaging log characterized by comparison with petrophysical properties and laboratory core data. In: SPE annual technical conference and exhibition. Society of Petroleum Engineers, Dallas, Texas, p. 9. <https://doi.org/10.2118/22723-MS>

[59] Coman R, Tietjen H, Thern H, Blanz M (2014). New large-hole magnetic resonance logging-while-drilling tool with short inter-echo time and improved vertical resolution. In: SPWLA 55th annual logging symposium.

[60] Coman R, Thern H, Kischkat T (2018). Lateral-motion correction of NMR logging-while-drilling data. In: SPWLA 59th annual logging symposium 2018.

[61] Connolly PRJ, Vogt SJ, Iglauer S, May EF, Johns ML (2017). Capillary trapping quantification in sandstones using NMR relaxometry. Water Resour Res 53:7917–7932. <https://doi.org/10.1002/2017WR020829>

[62] Connolly PRJ, Yan W, Zhang D, Mahmoud M, Verrall M, Lebedev M, Iglauer S, Metaxas PJ, May EF, Johns ML (2019). Simulation and experimental measurements of internal magnetic field gradients and NMR transverse relaxation times (T2) in sandstone rocks. J Pet Sci Eng 175:985–997. <https://doi.org/10.1016/j.petrol.2019.01.036>

[63] Cotts R, Hoch MJ, Sun T, Markert J (1989). Pulsed field gradient stimulated echo methods for improved NMR diffusion measurements in heterogeneous systems. J Magn Reson 83:252–266. <https://doi.org/10.1016/0022-2364(89)90189-3>

[64] Cozzolino S, Sanna MG, Valentini M (2008). Probing interactions by means of pulsed field gradient nuclear magnetic resonance spectroscopy. Magn Reson Chem 46:S16–S23. <https://doi.org/10.1002/mrc.2345>

[65] Curtis ME, Cardott BJ, Sondergeld CH, Rai CS (2012). Development of organic porosity in the Woodford Shale with increasing thermal maturity. Int J Coal Geol 103:26–31. <https://doi.org/10.1016/j.coal.2012.08.004>

[66] D'Agostino C, Bräuer P, Charoen-Rajapark P, Crouch MD, Gladden LF (2017). Effect of paramagnetic species on T1, T2 and T1/T2 NMR relaxation times of liquids in porous CuSO4/Al2O3. RSC Adv 7:36163–36167. <https://doi.org/10.1039/C7RA07165E>

[67] Dang ST, Sondergeld CH, Rai CS (2019). Interpretation of nuclear-magnetic-resonance response to hydrocarbons: application to miscible enhanced-oil-recovery experiments in shales. SPE Reserv Eval Eng 22:302–309. <https://doi.org/10.2118/191144-PA>

[68] Davis CR, Martinez CJ, Howarter JA, Erk KA (2021). Impact of saltwater environments on the coalescence of oil-in-water emulsions stabilized by an anionic surfactant. ACS ES&T Water 1:1702–1713. <https://doi.org/10.1021/acsestwater.1c00066>

[69] Demas V, Prado PJ, Hürlimann MD, Song YQ, Fantazzini P, Bortolotti V (2008). Compact magnets for magnetic resonance. In: AIP conference proceedings. AIP, pp. 36–39. <https://doi.org/10.1063/1.3058541>

[70] DePavia L, Heaton N, Ayers D, Freedman R, Harris R, Jorion B, Kovats J, Luong B, Rajan N, Taherian R, Walter K, Willis D, Scheibal J, Garcia S (2003). A next-generation wireline NMR logging tool. In: All Days. SPE, Denver, Colorado, p. 7.

[71] Diehl B (2008). Principles in NMR spectroscopy. In: Holzgrabe U, Wawer I, Diehl BBT (eds) NMR spectroscopy in pharmaceutical analysis. Elsevier, Amsterdam, pp 1–41.

[72] Doğan M, Göksel Saraç M, Aslan Türker D (2020). Effect of salt on the inter-relationship between the morphological, emulsifying and interfacial rheological properties of O/W emulsions at oil/water interface. J Food Eng 275:109871. <https://doi.org/10.1016/j.jfoodeng.2019.109871>

[73] Dong X, Shen LW, Liu X, Zhang P, Sun Y, Yan W, Jiang L, Wang F, Sun J (2020). NMR characterization of a tight sand's pore structures and fluid mobility: an experimental investigation for CO2 EOR potential. Mar Pet Geol 118:104460. <https://doi.org/10.1016/j.marpetgeo.2020.104460>

[74] Doughty DA, Tomutsa L (1996). Multinuclear NMR microscopy of two-phase fluid systems in porous rock. Magn Reson Imaging 14:869–873. <https://doi.org/10.1016/S0730-725X(96)00218-4>

[75] Ellis DV, Singer JM (2007). Well logging for earth scientists. Springer.

[76] Elsayed M, Glatz G, El-Husseiny A, Alqubalee A, Adebayo A, Al-Garadi K, Mahmoud M (2020a). The effect of clay content on the spin-spin NMR relaxation time measured in porous media. ACS Omega 5:6545–6555. <https://doi.org/10.1021/acsomega.9b04228>

[77] Elsayed M, Mahmoud M, El-Husseiny A, Kamal MS, Al-Garadi K (2020b). A new method to evaluate reaction kinetics of acids with carbonate rocks using NMR diffusion measurements. Energy Fuels 34:787–797. <https://doi.org/10.1021/acs.energyfuels.9b03784>

[78] Elsayed M, El-Husseiny A, Kadafur I, Mahmoud M, Aljawad MS, Alqubalee A (2021a). An experimental study on the effect of magnetic field strength and internal gradient on NMR-Derived petrophysical properties of sandstones. J Pet Sci Eng 205:108811. <https://doi.org/10.1016/j.petrol.2021.108811>

[79] Elsayed M, El-Husseiny A, Kwak H, Hussaini SR, Mahmoud M (2021b). New technique for evaluating fracture geometry and preferential orientation using pulsed field gradient nuclear magnetic resonance. SPE J. <https://doi.org/10.2118/205505-PA>

[80] Enwere PM, Archer JS (1992). NMR imaging for water/oil displacement in cores under viscous-capillary force control. In: SPE/DOE enhanced oil recovery symposium. Society of Petroleum Engineers. <https://doi.org/10.2118/24166-MS>

[81] Eslami M, Kadkhodaie-Ilkhchi A, Sharghi Y, Golsanami N (2013). Construction of synthetic capillary pressure curves from the joint use of NMR log data and conventional well logs. J Pet Sci Eng 111:50–58. <https://doi.org/10.1016/j.petrol.2013.10.010>

[82] Espitalie J, Madec M, Tissot B, Mennig JJ, Leplat P (1977). Source rock characterization method for petroleum exploration. In: offshore technology conference. <https://doi.org/10.4043/2935-MS>

[83] Faurissoux P, Colombain A, Pujol G, Fraute O, Nicot B (2018). Ultra-Fast Capillary Pressure and Resistivity measurements. In: RDPETRO 2018: research and development petroleum conference and exhibition, Abu Dhabi, UAE, 9–10 May 2018. American association of petroleum geologists, society of exploration geophysicists, European association of geoscientists and engineers, and society of petroleum engineers, pp. 132–135. <https://doi.org/10.1190/RDP2018-40973355.1>

[84] Flaum M, Chen J, Hirasaki GJ (2005). NMR diffusion editing for D–T2 maps: application to recognition of wettability change. Petrophysics 46:113–123.

[85] Fleury M, Deflandre F (2003). Quantitative evaluation of porous media wettability using NMR relaxometry. Magn Reson Imaging 21:385–387. <https://doi.org/10.1016/S0730-725X(03)00145-0>

[86] Fleury M, Romero-Sarmiento M (2016). Characterization of shales using T1–T2 NMR maps. J Pet Sci Eng 137:55–62. <https://doi.org/10.1016/j.petrol.2015.11.006>

[87] Foley I, Farooqui SA, Kleinberg RL (1996). Effect of paramagnetic ions on NMR relaxation of fluids at solid surfaces. J Magn Reson Ser A 123:95–104. <https://doi.org/10.1006/jmra.1996.0218>

[88] Fordham EJ, Mitchell J (2018). Localization in a single pore. Microporous Mesoporous Mater 269:35–38. <https://doi.org/10.1016/j.micromeso.2017.05.029>

[89] Fordham EJ, Horsfield MA, Hall LD, Maitland GC (1993). Depth filtration of clay in rock cores observed by one-dimensional 1H NMR imaging. J Colloid Interface Sci 156:253–255. <https://doi.org/10.1006/jcis.1993.1106>

[90] Freedman R, Heaton N (2004). Fluid characterization using nuclear magnetic resonance logging. Petrophysics 45:241–250.

[91] Freedman R, Lo S, Flaum M, Hirasaki GJ, Matteson A, Sezginer A (2001). A new NMR method of fluid characterization in reservoir rocks: experimental confirmation and simulation results. SPE J 6:452–464. <https://doi.org/10.2118/75325-PA>

[92] Freedman R, Heaton N, Flaum M, Hirasaki GJ, Flaum C, Hürlimann M (2003). Wettability, saturation, and viscosity from NMR measurements. SPE J 8:317–327. <https://doi.org/10.2118/87340-PA>

[93] Gamal H, Elkatatny S, Adebayo A (2021). Influence of mud filtrate on the pore system of different sandstone rocks. J Pet Sci Eng 202:108595. <https://doi.org/10.1016/j.petrol.2021.108595>

[94] Ge X, Myers MT, Liu J, Fan Y, Zahid MA, Zhao J, Hathon L (2021). Determining the transverse surface relaxivity of reservoir rocks: a critical review and perspective. Mar Pet Geol 126:104934. <https://doi.org/10.1016/j.marpetgeo.2021.104934>

[95] Ghomeshi S, Kryuchkov S, Kantzas A (2018). An investigation into the effects of pore connectivity on T NMR relaxation. J Magn Reson 289:79–91. <https://doi.org/10.1016/j.jmr.2018.02.007>

[96] Gladden LF, Mitchell J (2011). Measuring adsorption, diffusion and flow in chemical engineering: applications of magnetic resonance to porous media. New J Phys 13:035001. <https://doi.org/10.1088/1367-2630/13/3/035001>

[97] Glorioso JC, Aguirre O, Piotti G, Mengual JF (2003). Deriving capillary pressure and water saturation from NMR transversal relaxation times. In: Proc. SPE Lat. Am. Caribb. Pet. Eng. Conf. 418–430. <https://doi.org/10.2523/81057-ms>

[98] Godefroy S, Korb J-P, Fleury M, Bryant RG (2001). Surface nuclear magnetic relaxation and dynamics of water and oil in macroporous media. Phys Rev E 64:021605. <https://doi.org/10.1103/PhysRevE.64.021605>

[99] Gomes JC (2014). Characterization and modeling of a thick carbonate transition zone. In: Proc. - SPE Annu. Tech. Conf. Exhib. 7, 5656–5670. <https://doi.org/10.2118/173480-stu>

[100] Gong H, Zhu C, Zhang Y, Li Z, San Q, Xu L, Li Y, Dong M, Hassanzadeh H (2020). Experimental evaluation on the oil saturation and movability in the organic and inorganic matter of shale. Energy Fuels 34:8063–8073. <https://doi.org/10.1021/acs.energyfuels.0c00831>

[101] Gonzalez V, Jones M, Taylor SE (2016). Spin-spin relaxation time investigation of oil/brine/sand systems. Kinetics, effects of salinity, and implications for wettability and bitumen recovery. Energy Fuels 30:844–853. <https://doi.org/10.1021/acs.energyfuels.5b02352>

[102] Goodarzi N, Bryan JL, Mai AT, Kantzas A (2005). Heavy oil fluid testing with conventional and novel techniques. In: All Days. SPE. <https://doi.org/10.2118/97803-MS>

[103] Gordon RE, Hanley PE, Shaw D (1982). Topical magnetic resonance. Prog Nucl Magn Reson Spectrosc 15:1–47. <https://doi.org/10.1016/0079-6565(82)80007-1>

[104] Grattoni CA, Moss AK, Muggeridge AH, Jing XD (2003). An improved technique for deriving drainage capillary pressure from NMR T2 distributions 1–12.

[105] Green D, Dick J, McAloon M (2008). Oil/water imbibition and drainage capillary pressure determined by MRI on a wide sampling of rocks. In: PROCEEDING SCA SCA2008–01.

[106] Grunewald E, Knight R (2009). A laboratory study of NMR relaxation times and pore coupling in heterogeneous media. Geophysics 74:E215–E221. <https://doi.org/10.1190/1.3223712>

[107] Grunewald E, Knight R (2011). A laboratory study of NMR relaxation times in unconsolidated heterogeneous sediments. Geophysics 76:G73–G83. <https://doi.org/10.1190/1.3581094>

[108] Guan H, Brougham D, Sorbie KS, Packer KJ (2002). Wettability effects in a sandstone reservoir and outcrop cores from NMR relaxation time distributions. J Pet Sci Eng 34:35–54. <https://doi.org/10.1016/S0920-4105(02)00151-1>

[109] Gummerson RJ, Hall C, Hoff WD, Hawkes R, Holland GN, Moore WS (1979). Unsaturated water flow within porous materials observed by NMR imaging. Nature 281:56–57. <https://doi.org/10.1038/281056a0>

[110] Guo J, Xie R, Xiao L, Liu M, Gao L (2019). Nuclear magnetic resonance T1–T2 spectra in heavy oil reservoirs. Energies 12:2415. <https://doi.org/10.3390/en12122415>

[111] Guo J-C, Zhou H-Y, Zeng J, Wang K-J, Lai J, Liu Y-X (2020). Advances in low-field nuclear magnetic resonance (NMR) technologies applied for characterization of pore space inside rocks: a critical review. Pet Sci 17:1281–1297. <https://doi.org/10.1007/s12182-020-00488-0>

[112] Habina I, Radzik N, Topór T, Krzyżak AT (2017). Insight into oil and gas-shales compounds signatures in low field 1H NMR and its application in porosity evaluation. Microporous Mesoporous Mater 252:37–49. <https://doi.org/10.1016/j.micromeso.2017.05.054>

[113] Hahn EL (1950a). Nuclear induction due to free larmor precession. Phys Rev 77:297–298. <https://doi.org/10.1103/PhysRev.77.297.2>

[114] Hahn EL (1950b). Spin echoes. Phys Rev 80:580–594. <https://doi.org/10.1103/PhysRev.80.580>

[115] Hassan A, Mahmoud M, Al-Majed A, Elsayed M, Al-Nakhli A, BaTaweel M (2020). Performance analysis of thermochemical fluids in removing the gas condensate from different gas formations. J Nat Gas Sci Eng. <https://doi.org/10.1016/j.jngse.2020.103333>

[116] Heaton N, Jain V, Boling B, Oliver D, Degrange J-M, Ferraris P, Hupp D, Prabawa H, Torres Ribeiro M, Vervest E, Stockden I (2012). New generation magnetic resonance while drilling. In: All Days. SPE. <https://doi.org/10.2118/160022-MS>

[117] Heidler R, Morriss C, Hoshun R (2003). Design and implementation of a new magnetic resonance tool for the while drilling environment. In: SPWLA 44th annual logging symposium.

[118] Herlinger R, Dos Santos BCC (2018). The impact of pore type on NMR T2 and micp in bioclastic carbonate reservoirs. In: SPWLA 59th annual logging symposium. 2018.

[119] Hidajat I, Mohanty KK, Flaum M, Hirasaki G (2004). Study of vuggy carbonates using NMR and X-ray CT scanning. SPE Reserv Eval Eng 7:365–377. <https://doi.org/10.2118/88995-PA>

[120] Hinedi ZR, Chang AC, Anderson MA, Borchardt DB (1997). Quantification of microporosity by nuclear magnetic resonance relaxation of water imbibed in porous media. Water Resour Res 33:2697–2704. <https://doi.org/10.1029/97WR02408>

[121] Hollingsworth KG, Johns ML (2003). Measurement of emulsion droplet sizes using PFG NMR and regularization methods. J Colloid Interface Sci 258:383–389. <https://doi.org/10.1016/S0021-9797(02)00131-5>

[122] Horkowitz J, Crary S, Ganesan K, Heidler R, Luong B, Morley J, Petricola M, Prusiecki C, Poitzsch M, Scheibal JR, Hashem M (2002). Applications of a new magnetic resonance logging-while-drilling tool in a gulf of mexico deepwater development project. In: SPWLA 43rd annual logging symposium.

[123] Hosseinzadeh S, Kadkhodaie A, Yarmohammadi S (2020). NMR derived capillary pressure and relative permeability curves as an aid in rock typing of carbonate reservoirs. J Pet Sci Eng 184:106593. <https://doi.org/10.1016/j.petrol.2019.106593>

[124] Hoult D, Richards R (1976). The signal-to-noise ratio of the nuclear magnetic resonance experiment. J Magn Reson 24:71–85. <https://doi.org/10.1016/0022-2364(76)90233-X>

[125] Howard JJ (1998). Quantitative estimates of porous media wettability from proton NMR measurements. Magn Reson Imaging 16:529–533. <https://doi.org/10.1016/S0730-725X(98)00060-5>

[126] Howard JJ, Kenyon WE (1992). Determination of pore size distribution in sedimentary rocks by proton nuclear magnetic resonance. Mar Pet Geol 9:139–145. <https://doi.org/10.1016/0264-8172(92)90086-T>

[127] Howard JJ, Kenyon WE, Straley C (1993). Proton magnetic resonance and pore size variations in reservoir sandstones. SPE Form Eval 8:194–200. <https://doi.org/10.2118/20600-PA>

[128] Howard JJ, Kenyon WE, Morriss CE, Straley C (1995). NMR in partially saturated rocks: laboratory insights on free fluid index and comparison with borehole logs. Log Anal 36.

[129] Hrabe J, Kaur G, Guilfoyle D (2007). Principles and limitations of NMR diffusion measurements. J Med Phys 32:34. <https://doi.org/10.4103/0971-6203.31148>

[130] Hu MD (1998). Effective gradients in porous media due to susceptibility differences. J Magn Reson 240:232–240.

[131] Hürlimann MD (1998). Effective gradients in porous media due to susceptibility differences. J Magn Reson 131:232–240. <https://doi.org/10.1006/jmre.1998.1364>

[132] Hürlimann MD, Heaton NJ (2015). NMR well logging. In: Johns ML, Fridjonnson EO, Vogt SJ, Haber A (eds) Mobile NMR and MRI: developments and applications. The Royal Society of Chemistry, Cambridge, pp 11–85.

[133] Hursan GG, Deering JS, Kelly FN (2015). NMR logs help formation testing and evaluation. In: All Days. SPE. <https://doi.org/10.2118/177974-MS>

[134] Hursan G, Silva A, Zeghlache ML (2016). Evaluation and development of complex clastic reservoirs using NMR. In: Paper presented at the SPE annual technical conference and exhibition, Dubai, UAE, September 2016. SPE. <https://doi.org/10.2118/181525-MS>

[135] Isah A, Adebayo AR, Mahmoud M, Babalola LO, El-Husseiny A (2021a). Drainage mechanisms in gas reservoirs with bimodal pores – a core and pore scale study. J Nat Gas Sci Eng 86:103652. <https://doi.org/10.1016/j.jngse.2020.103652>

[136] Isah A, Adebayo AR, Mahmoud M, Babalola LO, El-Husseiny A (2021b). Characterization of fluid drainage mechanism at core and pore scales: an NMR capillary pressure–based saturation exponent prediction. In: Day 4 Thu, October 21, 2021. SPE. <https://doi.org/10.2118/205176-MS>

[137] Jachmann R, Yang J, Wang Y (2020). Motion artifact free data delivery in real time from a low gradient nmr tool. In: SPWLA 61st annual logging symposium. <https://doi.org/10.30632/SPWLA-5014>

[138] Jackson JA, Burnett LJ, Harmon JF (1980). Remote (inside-out) NMR. III. Detection of nuclear magnetic resonance in a remotely produced region of homogeneous magnetic field. J Magn Reson 41:411–421. <https://doi.org/10.1016/0022-2364(80)90298-X>

[139] Jaeger F, Bowe S, Van As H, Schaumann GE (2009). Evaluation of 1H NMR relaxometry for the assessment of pore-size distribution in soil samples. Eur J Soil Sci 60:1052–1064. <https://doi.org/10.1111/j.1365-2389.2009.01192.x>

[140] Ji Y, Hou J, Zhao E, Lu N, Bai Y, Zhou K, Liu Y (2020). Study on the effects of heterogeneous distribution of methane hydrate on permeability of porous media using low-field NMR technique. J Geophys Res Solid Earth. <https://doi.org/10.1029/2019JB018572>

[141] Johns ML (2009). NMR studies of emulsions. Curr Opin Colloid Interface Sci 14:178–183. <https://doi.org/10.1016/j.cocis.2008.10.005>

[142] Johns ML, Gladden LF (2002). Sizing of emulsion droplets under flow using flow-compensating NMR-PFG techniques. J Magn Reson 154:142–145. <https://doi.org/10.1006/jmre.2001.2469>

[143] Johns ML, Fridjonsson EO, Vogt SJ, Haber A (eds) (2015). Mobile NMR and MRI, new developments in NMR. Royal Society of Chemistry, Cambridge.

[144] Johnson CS (1999). Diffusion ordered nuclear magnetic resonance spectroscopy: principles and applications. Prog Nucl Magn Reson Spectrosc 34:203–256. <https://doi.org/10.1016/S0079-6565(99)00003-5>

[145] Johnson DL, Schwartz LM (2014). Analytic theory of two-dimensional NMR in systems with coupled macro- and micropores. Phys Rev E 90:032407. <https://doi.org/10.1103/PhysRevE.90.032407>

[146] Kanfar MF (2012). Real-time integrated petrophysics: geosteering in challenging geology and fluid systems. In: Soc. Pet. Eng. - SPE Saudi Arab. Sect. Young Prof. Tech. Symp. 2012, YPTS 2012 45–54. <https://doi.org/10.2118/160922-ms>

[147] Karlsons K, de Kort DW, Sederman AJ, Mantle MD, Freeman JJ, Appel M, Gladden LF (2021). Characterizing pore-scale structure-flow correlations in sedimentary rocks using magnetic resonance imaging. Phys Rev E 103:023104. <https://doi.org/10.1103/PhysRevE.103.023104>

[148] Katika T, Saidian M, Prasad M, Fabricius I (2017). Low-field NMR spectrometry of chalk and argillaceous sandstones: rock-fluid affinity assessed from T1/T2 ratio. Petrophysics 58:126–140.

[149] Kausik R, Fellah K, Rylander E, Singer PM, Lewis RE, Sinclair SM (2016). NMR relaxometry in shale and implications for logging. J Petrophys SPWLA J Form Eval Reserv Descr 57:339–350.

[150] Kausik R, Fellah K, Feng L, Simpson G (2017). High- and low-field NMR relaxometry and diffusometry of the bakken petroleum system. J Petrophys- SPWLA J Form Eval Reserv Descr 58:341–351.

[151] Keating K, Knight R (2007). A laboratory study to determine the effect of iron oxides on proton NMR measurements. Geophysics 72:E27–E32. <https://doi.org/10.1190/1.2399445>

[152] Keating K, Knight R, Tufano KJ (2008). Nuclear magnetic resonance relaxation measurements as a means of monitoring iron mineralization processes. Geophys Res Lett 35:L19405. <https://doi.org/10.1029/2008GL035225>

[153] Kenyon WE (1997). Petrophysical principles of applications of NMR logging. Log Anal 38:23.

[154] Kenyon WE, Day PI, Straley C, Willemsen JF (1988). A three-part study of NMR longitudinal relaxation properties of water-saturated sandstones. SPE Form Eval 3:622–636. <https://doi.org/10.2118/15643-PA>

[155] Kleinberg RL, Jackson JA (2001). An introduction to the history of NMR well logging. Concepts Magn Reson 13:340–342. <https://doi.org/10.1002/cmr.1018>

[156] Kleinberg RL, Kenyon WE, Mitra PP (1994). Mechanism of NMR relaxation of fluids in rock. J Magn Reson Ser A 108:206–214. <https://doi.org/10.1006/jmra.1994.1112>

[157] Kleinberg RL, Griffin DD, Fukuhara M, Sezginer A, Chew WC, Kenyon WE, Day PI, Lipsicas M (1990). Borehole measurement of NMR characteristics of earth formations, and interpretations thereof.

[158] Kleinberg RL, Straley C, Kenyon WE, Akkurt R, Farooqui SA (1993). Nuclear magnetic resonance of rocks: T1 vs T2. In: Proc. - SPE Annu. Tech. Conf. Exhib. Omega, 553–563. <https://doi.org/10.2523/26470-ms>

[159] Knight R, Walsh DO, Butler JJ Jr, Grunewald E, Liu G, Parsekian AD, Reboulet EC, Knobbe S, Barrows M (2016). NMR logging to estimate hydraulic conductivity in unconsolidated aquifers. Groundwater 54(1):104–114. <https://doi.org/10.1111/gwat.12324>

[160] Korb J-P, Xu S, Jonas J (1993). Confinement effects on dipolar relaxation by translational dynamics of liquids in porous silica glasses. J Chem Phys 98:2411–2422. <https://doi.org/10.1063/1.464169>

[161] Kruspe T, Thern HF, Kurz G, Blanz M, Akkurt R, Ruwaili S, Seifert D, Marsala AF (2009). Slimhole application of magnetic resonance while drilling. In: SPWLA 50th annual logging symposium.

[162] Kwak HT, Wang J, AlSofi AM (2017). Close monitoring of gel based conformance control by NMR techniques. In: Day 2 Tue, March 07, 2017. SPE. <https://doi.org/10.2118/183719-MS>

[163] Ladd ME, Bachert P, Meyerspeer M, Moser E, Nagel AM, Norris DG, Schmitter S, Speck O, Straub S, Zaiss M (2018). Pros and cons of ultra-high-field MRI/MRS for human application. Prog Nucl Magn Reson Spectrosc 109:1–50. <https://doi.org/10.1016/j.pnmrs.2018.06.001>

[164] Lai J, Wang K, Zhou H, Zhao J, Wu L (2020). Variation of limestone pore structure under acidizing and wormhole propagation visualization using NMR. In: SPE international conference and exhibition on formation damage control. Society of Petroleum Engineers. <https://doi.org/10.2118/199327-MS>

[165] Lalanne B, Rebelle M (2014). A review of alternative methods to classify rock-types from capillary pressure measurements. In: All Days. IPTC. <https://doi.org/10.2523/IPTC-17631-MS>

[166] Lawal LO, Adebayo AR, Mahmoud M, Dia BM, Sultan AS (2020). A novel NMR surface relaxivity measurements on rock cuttings for conventional and unconventional reservoirs. Int J Coal Geol 231:103605. <https://doi.org/10.1016/j.coal.2020.103605>

[167] Levitt MH (2013). Spin dynamics: basics of nuclear magnetic resonance. Wiley, London.

[168] Li M, Romero-Zerón L, Marica F, Balcom BJ (2017). Polymer flooding enhanced oil recovery evaluated with magnetic resonance imaging and relaxation time measurements. Energy Fuels 31:4904–4914. <https://doi.org/10.1021/acs.energyfuels.7b00030>

[169] Li Z, Mao Z, Sun Z, Luo X, Wang Z, Zhao P (2019). An NMR-based clay content evaluation method for tight oil reservoirs. J Geophys Eng 16:116–124. <https://doi.org/10.1093/jge/gxy010>

[170] Liang C, Xiao L, Zhou C, Wang H, Hu F, Liao G, Jia Z, Liu H (2019). Wettability characterization of low-permeability reservoirs using nuclear magnetic resonance: an experimental study. J Pet Sci Eng 178:121–132. <https://doi.org/10.1016/j.petrol.2019.03.014>

[171] Liaw H-K, Kulkarni R, Chen S, Watson AT (1996). Characterization of fluid distributions in porous media by NMR techniques. AIChE J 42:538–546. <https://doi.org/10.1002/aic.690420223>

[172] Lin X, Ruan R, Chen P, Chung M, Ye X, Yang T, Doona C, Wagner T (2006). NMR state diagram concept. J Food Sci 71:R136–R145. <https://doi.org/10.1111/j.1750-3841.2006.00193.x>

[173] Ling NNA, Haber A, Graham BF, Aman ZM, May EF, Fridjonsson EO, Johns ML (2018). Quantifying the effect of salinity on oilfield water-in-oil emulsion stability. Energy Fuels 32:10042–10049. <https://doi.org/10.1021/acs.energyfuels.8b02143>

[174] Liu H (2017). Principles and applications of well logging, all days. Springer, Berlin.

[175] Livo K, Saidian M, Prasad M (2020). Effect of paramagnetic mineral content and distribution on nuclear magnetic resonance surface relaxivity in organic-rich Niobrara and Haynesville shales. Fuel 269:117417. <https://doi.org/10.1016/j.fuel.2020.117417>

[176] Looyestijn WJ (2008). Wettability index determination from NMR logs. Petrophysics 49:16.

[177] Luo Z-X, Paulsen J, Song Y-Q (2015). Robust determination of surface relaxivity from nuclear magnetic resonance DT2 measurements. J Magn Reson 259:146–152. <https://doi.org/10.1016/j.jmr.2015.08.002>

[178] Lyngra S, Hursan GG, Palmer RG, Zeybek M, Ayyad HA, Qureshi A (2015). Heavy oil characterization: lessons learned during placement of a horizontal injector at a tar/oil interface. In: All Days. SPE. <https://doi.org/10.2118/172673-MS>

[179] Mai A, Kantzas A (2007). Porosity distributions in carbonate reservoirs using low-field NMR. J Can Pet Technol. <https://doi.org/10.2118/07-07-02>

[180] Makeen YM, Abdullah WH, Pearson MJ, Hakimi MH, Elhassan OMA, Hadad YT (2016). Thermal maturity history and petroleum generation modelling for the Lower Cretaceous Abu Gabra Formation in the Fula Sub-basin, Muglad Basin, Sudan. Mar Pet Geol 75:310–324. <https://doi.org/10.1016/j.marpetgeo.2016.04.023>

[181] Makeen YM, Shan X, Lawal M, Ayinla HA, Su S, Yelwa NA, Liang Y, Ayuk NE, Du X (2021). Reservoir quality and its controlling diagenetic factors in the Bentiu Formation, Northeastern Muglad Basin, Sudan. Sci Rep 11:18442. <https://doi.org/10.1038/s41598-021-97994-x>

[182] Mamoudou S, Tinni A, Curtis M, Sondergeld CH, Rai CS (2021). Impact of EOR Huff-n-puff on rock microstructure. In: SPE/AAPG/SEG Unconv Resources Technology Conference. <https://doi.org/10.15530/urtec-2021-5664>

[183] Mankinen O, Zhivonitko VV, Selent A, Mailhiot S, Komulainen S, Prisle NL, Ahola S, Telkki V-V (2020). Ultrafast diffusion exchange nuclear magnetic resonance. Nat Commun 11:3251. <https://doi.org/10.1038/s41467-020-17079-7>

[184] Markovic S, Bryan JL, Turakhanov A, Cheremisin A, Mehta SA, Kantzas A (2020). In-situ heavy oil viscosity prediction at high temperatures using low-field NMR relaxometry and nonlinear least squares. Fuel 260:116328. <https://doi.org/10.1016/j.fuel.2019.116328>

[185] Martinez GA, Davis LA (2000). Petrophysical measurements on shales using NMR. In: All Days. SPE. <https://doi.org/10.2118/62851-MS>

[186] Mazumder A, Dubey DK (2013). Nuclear magnetic resonance (NMR) spectroscopy. In: reference module in chemistry, molecular sciences and chemical engineering. Elsevier. <https://doi.org/10.1016/B978-0-12-409547-2.05891-1>

[187] McCall KR, Johnson DL, Guyer RA (1991). Magnetization evolution in connected pore systems. Phys Rev B 44:7344–7355. <https://doi.org/10.1103/PhysRevB.44.7344>

[188] Meiboom S, Gill D (1958). Modified spin-echo method for measuring nuclear relaxation times. Rev Sci Instrum 29:688–691. <https://doi.org/10.1063/1.1716296>

[189] Mendelson KS (1985). Nuclear magnetic relaxation in porous media. In: proceedings - the electrochemical society. Electrochemical soc, pp. 282–291. <https://doi.org/10.1149/1.2108633>

[190] Minh CC, Crary S, Singer PM, Valori A, Bachman N, Hursan G, Ma S, Belowi A, Kraishan G, Aramco S (2015). Determination of wettability from magnetic resonance relaxation and diffusion measurements on fresh state cores. In: SPWLA 56th annual logging symposium, July 18–22, 2015. Society of Petrophysicists and Well-Log Analysts.

[191] Mitchell J, Griffith JD, Collins JHP, Sederman AJ, Gladden LF, Johns ML (2007). Validation of NMR relaxation exchange time measurements in porous media. J Chem Phys 127:234701. <https://doi.org/10.1063/1.2806178>

[192] Mitchell J, Chandrasekera TC, Johns ML, Gladden LF, Fordham EJ (2010). Nuclear magnetic resonance relaxation and diffusion in the presence of internal gradients: the effect of magnetic field strength. Phys Rev E 81:1–19. <https://doi.org/10.1103/PhysRevE.81.026101>

[193] Mitchell J, Chandrasekera TC, Gladden LF (2012a). Numerical estimation of relaxation and diffusion distributions in two dimensions. Prog Nucl Magn Reson Spectrosc 62:34–50. <https://doi.org/10.1016/j.pnmrs.2011.07.002>

[194] Mitchell J, Staniland J, Chassagne R, Fordham EJ (2012c). Quantitative in situ enhanced oil recovery monitoring using nuclear magnetic resonance. Transp Porous Media 94:683–706. <https://doi.org/10.1007/s11242-012-0019-8>

[195] Mitchell J, Chandrasekera TC, Holland DJ, Gladden LF, Fordham EJ (2013). Magnetic resonance imaging in laboratory petrophysical core analysis. Phys Rep 526:165–225. <https://doi.org/10.1016/j.physrep.2013.01.003>

[196] Mitchell J, Gladden LF, Chandrasekera TC, Fordham EJ (2014a). Low-field permanent magnets for industrial process and quality control. Prog Nucl Magn Reson Spectrosc 76:1–60. <https://doi.org/10.1016/j.pnmrs.2013.09.001>

[197] Mitchell J, Howe AM, Clarke A (2015). Real-time oil-saturation monitoring in rock cores with low-field NMR. J Magn Reson 256:34–42. <https://doi.org/10.1016/j.jmr.2015.04.011>

[198] Mitchell J, Souza A, Fordham E, Boyd A (2019). A finite element approach to forward modeling of nuclear magnetic resonance measurements in coupled pore systems. J Chem Phys 150:154708. <https://doi.org/10.1063/1.5092159>

[199] Mitchell J, Fordham EJ (2014). Contributed Review: Nuclear magnetic resonance core analysis at 0.3 T. Rev Sci Instrum 85:111502. <https://doi.org/10.1063/1.4902093>

[200] Mitchell J, Edwards J, Fordham E, Staniland J, Chassagne R, Cherukupalli P, Wilson O, Faber R, Bouwmeester R (2012a). Quantitative remaining oil interpretation using magnetic resonance: from the laboratory to the pilot. In: All Days. SPE. <https://doi.org/10.2118/154704-MS>

[201] Mitchell J, Staniland J, Wilson A, Howe A, Clarke A, Fordham EJ, Edwards J, Faber R, Bouwmeester R (2012c). Magnetic resonance imaging of chemical EOR in core to complement field pilot studies. In: international symposium social core anal. Aberdeen Scotland, UK, 27–30 August 2012e. SCA Paper 2012e–30.

[202] Mitchell J, Staniland J, Wilson A, Howe A, Clarke A, Fordham EJ, Edwards J, Faber R, Bouwmeester R (2014a). Monitoring chemical EOR processes. In: All Days. SPE. <https://doi.org/10.2118/169155-MS>

[203] Mitchell J (2016). Industrial applications of magnetic resonance diffusion and relaxation time measurements. In: Valiullin R (ed) Diffusion NMR of confined systems: fluid transport in porous solids and heterogeneous materials. The Royal Society of Chemistry, Cambridge, pp 353–389.

[204] Mohnke O, Jorand R, Nordlund C, Klitzsch N (2015). Understanding NMR relaxometry of partially water-saturated rocks. Hydrol Earth Syst Sci 19:2763–2773. <https://doi.org/10.5194/hess-19-2763-2015>

[205] Morales O, Sierra F, Hazboun N, Saucedo G, Jain V, Bachman N, Gzara K (2016). Identifying a bypassed reservoir in limestone "A" Sequence With Magnetic Resonance While Drilling 41843.

[206] Morley J, Heidler R, Horkowitz J, Luong B, Woodburn C, Poitzsch M, Borbas T, Wendt B (2002). Field testing of a new nuclear magnetic resonance logging-while-drilling tool. In: All Days. SPE, pp. 1273–1284. <https://doi.org/10.2118/77477-MS>

[207] Moser E, Laistler E, Schmitt F, Kontaxis G (2017). Ultra-high field NMR and MRI—the role of magnet technology to increase sensitivity and specificity. Front Phys 5:33. <https://doi.org/10.3389/fphy.2017.00033>

[208] Mukhametdinova A, Mikhailova P, Kozlova E, Karamov T, Baluev A, Cheremisin A (2020). Effect of thermal exposure on oil shale saturation and reservoir properties. Appl Sci 10:9065. <https://doi.org/10.3390/app10249065>

[209] Neuringer LJ (1990). Nuclear magnetic resonance spectroscopy and imaging of humans. Phys B Condens Matter 164:193–199. <https://doi.org/10.1016/0921-4526(90)90075-6>

[210] Newgord C, Tandon S, Heidari Z (2020). Simultaneous assessment of wettability and water saturation using 2D NMR measurements. Fuel 270:117431. <https://doi.org/10.1016/j.fuel.2020.117431>

[211] Nicolay K, Braun KPJ, de Graaf RA, Dijkhuizen RM, Kruiskamp MJ (2001). Diffusion NMR spectroscopy. NMR Biomed 14:94–111. <https://doi.org/10.1002/nbm.686>

[212] Nørgaard JV, Olsen D, Reffstrup J, Springer N (1999). Capillary-pressure curves for low-permeability chalk obtained by nuclear magnetic resonance imaging of core-saturation profiles. SPE Reserv Eval Eng 2:141–148. <https://doi.org/10.2118/55985-PA>

[213] Oguntona JA, Kelsch K, Osman K, Ingebrigtsen E, Butt P, Saha S (2004). Thin sand development made possible through enhanced geosteering and reservoir planning with while-drilling resistivity and NMR logs: example from Niger delta. In: Soc. Pet. Eng. - Niger. Annu. Int. Conf. Exhib. 2004, NAICE 2004. <https://doi.org/10.2118/88889-ms>

[214] Ouellette M, Li M, Liao G, Hussein EMA, Romero-Zeron L, Balcom BJ (2015). Rock core analysis: metallic core holders for magnetic resonance imaging under reservoir conditions. In: mobile NMR and MRI: developments and applications. The Royal Society of Chemistry, pp. 290–309. <https://doi.org/10.1039/9781782628095-00290>

[215] Pan J, Liao G, Su R, Chen S, Wang Z, Chen L, Chen L, Wang X, Guo Y (2021). 13C solid-state NMR analysis of the chemical structure in petroleum coke during idealized in situ combustion conditions. ACS Omega 6:15479–15485. <https://doi.org/10.1021/acsomega.1c02055>

[216] Panattoni F, Mitchell J, Fordham EJ, Kausik R, Grey CP, Magusin PCMM (2021b). Combined high-resolution solid-state 1H/13C NMR spectroscopy and 1H NMR relaxometry for the characterization of kerogen thermal maturation. Energy Fuels 35:1070–1079. <https://doi.org/10.1021/acs.energyfuels.0c02713>

[217] Panattoni F, Colbourne AA, Fordham EJ, Mitchell J, Grey CP, Magusin PCMM (2021). Improved description of organic matter in shales by enhanced solid fraction detection with low-field 1H NMR relaxometry. Energy Fuels. <https://doi.org/10.1021/acs.energyfuels.1c02386>

[218] Passey QR, Bohacs KM, Esch WL, Klimentidis R, Sinha S (2010). From oil-prone source rock to gas-producing shale reservoir – geologic and petrophysical characterization of unconventional shale-gas reservoirs. In: All Days. SPE. <https://doi.org/10.2118/131350-MS>

[219] Peña AA, Hirasaki GJ, Miller CA (2005). Chemically induced destabilization of water-in-crude oil emulsions. Ind Eng Chem Res 44:1139–1149. <https://doi.org/10.1021/ie049666i>

[220] Peng L, Zhang C, Ma H, Pan H (2019). Estimating irreducible water saturation and permeability of sandstones from nuclear magnetic resonance measurements by fractal analysis. Mar Pet Geol 110:565–574. <https://doi.org/10.1016/j.marpetgeo.2019.07.037>

[221] Pires LO, Winter A, Trevisan OV (2019). Dolomite cores evaluated by NMR. J Pet Sci Eng 176:1187–1197. <https://doi.org/10.1016/j.petrol.2018.06.026>

[222] Prammer MG, Drack E, Goodman G, Masak P, Menger S, Morys M, Zannoni S, Suddarth B, Dudley J (2000a). The magnetic resonance while-drilling tool: theory and operation. In: proceedings of SPE annual technical conference and exhibition. Society of Petroleum Engineers, pp. 281–288. <https://doi.org/10.2523/62981-MS>

[223] Prammer MG, Goodman GD, Menger SK, Morys M, Zannoni S, Dudley JH (2000b). Field test of an experimental NMR LWD Device. In: SPWLA 41st annual logging symposium.

[224] Prammer MG, Akkurt R, Cherry R, Menger S (2002). A new direction in wireline and LWD NMR. In: SPWLA 43rd annual logging symposium.

[225] Price WS (1997). Pulsed-field gradient nuclear magnetic resonance as a tool for studying translational diffusion: Part 1. Basic Theory. Concepts Magn Reson 9:299–336.

[226] Price WS (2009). NMR studies of translational motion. Cambridge University Press, Cambridge. <https://doi.org/10.1017/CBO9780511770487>

[227] Radwan AE, Trippetta F, Kassem AA, Kania M (2021). Multi-scale characterization of unconventional tight carbonate reservoir: insights from October oil filed, Gulf of Suez rift basin. Egypt. J Petrol Sci Eng 197:107968.

[228] Raheem ON, Fernandes MO, Thomas NC, Hashem MH, Alfazazi U, Sulemana NT (2017). Using NMR T2 to predict the drainage capillary curves Pc-Sw in carbonates reservoirs. In: Soc. Pet. Eng. - SPE Reserv. Characterisation Simul. Conf. Exhib. RCSC 2017 1–34. <https://doi.org/10.3997/2214-4609.201702465>

[229] Rezaee R, Saeedi A, Clennell B (2012). Tight gas sands permeability estimation from mercury injection capillary pressure and nuclear magnetic resonance data. J Pet Sci Eng 88–89:92–99. <https://doi.org/10.1016/j.petrol.2011.12.014>

[230] Ribeiro M, Costa V, Guedes R, Bittencourt P, Ferraris P, Guedes A (2012). Integrated petrophysics and geosteering reservoir characterization in the initial development phase of a carbonate reservoir - Campos Basin, Offshore Brazil. In: SPE Lat. Am. Caribb. Pet. Eng. Conf. Proc. 1, 1–13. <https://doi.org/10.4043/22738-ms>

[231] Richardson I (1999). The nature of C-S-H in hardened cements. Cem Concr Res 29:1131–1147. <https://doi.org/10.1016/S0008-8846(99)00168-4>

[232] Rios EH, Figueiredo I, Moss AK, Pritchard TN, Glassborow BA, Domingues ABG, de Azeredo RBV (2016). NMR permeability estimators in 'chalk' carbonate rocks obtained under different relaxation times and MICP size scalings. Geophys J Int 206:260–274. <https://doi.org/10.1093/gji/ggw130>

[233] Rueslåtten H, Eidesmo T, Slot-Petersen C (1998). NMR studies of an iron-rich sandstone oil reservoir. In: proceeding 1998.

[234] Sakthivel S, Elsayed M (2021). Enhanced oil recovery by spontaneous imbibition of imidazolium based ionic liquids on the carbonate reservoir. J Mol Liq. <https://doi.org/10.1016/j.molliq.2021.117301>

[235] Salomon Marques D, White R, Al-Khabaz S, Al-Talaq M, Al-Buainain J (2020). Benchmarking of pulsed field gradient nuclear magnetic resonance as a demulsifier selection tool with Arabian light crude oils. SPE Prod Oper Preprint. <https://doi.org/10.2118/203820-PA>

[236] Seevers DO (1966). A nuclear magnetic method for determining the permeability of sandstones. SPWLA 7th annual logging symposium.

[237] Seifert DJ, Akkurt R, Al-Dossary S, Shokeir R, Ersoz H (2007). Nuclear magnetic resonance logging: while drilling, wireline, and fluid sampling. In: SPE middle east oil gas show conference MEOS, proceedings 3, 1312–1321. <https://doi.org/10.2118/105605-ms>

[238] Selheim NB, Morris SA, Jonsbraaten F, Aarnes I, Teelken R (2017). Geosteering and mapping of complex reservoir boundaries using an integrated data approach. In: proceedings- SPE annual technology conference exhibition. <https://doi.org/10.2118/187136-ms>

[239] Serry AM, Herz U, Tagarieva L (2016). Reservoir characterization while drilling; a real time geosteering answer to maximize well values. A case study, Offshore Abu Dhabi. In: Day 1 Mon, November 07, 2016. SPE. <https://doi.org/10.2118/183092-MS>

[240] Shikhov I, Li R, Arns CH (2018). Relaxation and relaxation exchange NMR to characterise asphaltene adsorption and wettability dynamics in siliceous systems. Fuel 220:692–705. <https://doi.org/10.1016/j.fuel.2018.02.059>

[241] Shikhov I, Thomas DS, Arns CH (2019). On the optimum aging time: magnetic resonance study of asphaltene adsorption dynamics in sandstone rock. Energy Fuels 33:8184–8201. <https://doi.org/10.1021/acs.energyfuels.9b01609>

[242] Silletta EV, Vila GS, Domené EA, Velasco MI, Bedini PC, Garro-Linck Y, Masiero D, Monti GA, Acosta RH (2022). Organic matter detection in shale reservoirs using a novel pulse sequence for T1–T2 relaxation maps at 2 MHz. Fuel 312:122863. <https://doi.org/10.1016/j.fuel.2021.122863>

[243] Singer P (2013). 1D and 2D NMR core-log integration in organic shale.

[244] Sjöblom J, Hemmingsen PV, Kallevik H (2017). The Role of asphaltenes in stabilizing water-in-crude oil emulsions. In: Mullins OC, Sheu EY, Hammami A, Marshell AG (eds) Asphaltenes, heavy oils, and petroleomics. Springer, New York, pp 549–587.

[245] Solatpour R, Kantzas A (2019). Application of nuclear magnetic resonance permeability models in tight reservoirs. Can J Chem Eng 97:1191–1207. <https://doi.org/10.1002/cjce.23354>

[246] Sondergeld CH, Ambrose RJ, Rai CS, Moncrieff J (2010). Microstructural studies of gas shales. In: All Days. SPE. <https://doi.org/10.2118/131771-MS>

[247] Song Y-Q (2007). Novel NMR techniques for porous media research. Cem Concr Res 37:325–328. <https://doi.org/10.1016/j.cemconres.2006.02.013>

[248] Song Y-Q, Kausik R (2019). NMR application in unconventional shale reservoirs – a new porous media research frontier. Prog Nucl Magn Reson Spectrosc 112–113:17–33. <https://doi.org/10.1016/j.pnmrs.2019.03.002>

[249] Song Y-Q, Carneiro G, Schwartz LM, Johnson DL (2014). Experimental identification of diffusive coupling using 2D NMR. Phys Rev Lett 113:235503. <https://doi.org/10.1103/PhysRevLett.113.235503>

[250] Stejskal EO, Tanner JE (1965). Spin diffusion measurements: spin echoes in the presence of a time-dependent field gradient. J Chem Phys 42:288–292. <https://doi.org/10.1063/1.1695690>

[251] Suekane T, Furukawa N, Tsushima S, Hirai S, Kiyota M (2009). Application of MRI in the measurement of two-phase flow of supercritical CO2 and water in porous rocks. J Porous Media 12:143–154. <https://doi.org/10.1615/JPorMedia.v12.i2.40>

[252] Sun Z, Xiao L, Zhang Y, Liao G, Xiang W, Tang L, Luo S, Liu W, Chen W, Tian Z, Hou X (2018). A modular and multi-functional single-sided NMR sensor. Microporous Mesoporous Mater 269:175–179. <https://doi.org/10.1016/j.micromeso.2017.05.039>

[253] Sun Z, Xiao L, Liao G, Li X, Hou X, Chen Z, Lu R (2020). Design of a new LWD NMR tool with high mechanical reliability. J Magn Reson 317:106791. <https://doi.org/10.1016/j.jmr.2020.106791>

[254] Sun J, Cai J, Feng P, Sun F, Li J, Lu J, Yan W (2021). Study on nuclear magnetic resonance logging T2 spectrum shape correction of sandstone reservoirs in oil-based mud wells. Molecules 26:6082. <https://doi.org/10.3390/molecules26196082>

[255] Sylta KE (2010). Primary drainage capillary pressure curves in heterogeneous carbonates with ultracentrifuge and NMR. (Master's thesis, Univ. Bergen).

[256] Tan M, Mao K, Song X, Yang X, Xu J (2015). NMR petrophysical interpretation method of gas shale based on core NMR experiment. J Pet Sci Eng 136:100–111. <https://doi.org/10.1016/j.petrol.2015.11.007>

[257] Tan M, Fan L, Mao K, Li J, Wu C (2019). Influential factors analysis and porosity correction method of nuclear magnetic resonance measurement in igneous rocks. J Appl Geophys 161:153–166. <https://doi.org/10.1016/j.jappgeo.2018.12.023>

[258] Tandon S, Heidari Z (2018). Effect of internal magnetic-field gradients on nuclear-magnetic-resonance measurements and nuclear-magnetic-resonance-based pore-network characterization. SPE Reserv Eval Eng 21:609–625. <https://doi.org/10.2118/181532-PA>

[259] Tandon S, Newgord C, Heidari Z (2020). Wettability quantification in mixed-wet rocks using a new NMR-based method. SPE Reserv Eval Eng Preprint. <https://doi.org/10.2118/191509-PA>

[260] Tandon S, Rostami A, Heidari Z (2017). A new NMR-based method for wettability assessment in mixed-wet rocks. In: Day 2 Tue, October 10, 2017. SPE. <https://doi.org/10.2118/187373-MS>

[261] Tester F, Deviated H, Rlo K, Wkdw S, Qrw G, Xvlqj ÀRZ, Uhvlgxh S, Lq S, Uh S, Ru Q (2009). Real-time detection of tar in carbonates using LWD triple combo, 50.

[262] Thrane LW, Seymour JD, Codd SL (2019). Probing diffusion dynamics during hydrate formation by high field NMR relaxometry and diffusometry. J Magn Reson 303:7–16. <https://doi.org/10.1016/j.jmr.2019.04.003>

[263] Tikhonov AN, Arsenin VY (1977). Solutions of ill-posed problems. Wiley, New York.

[264] Timur A (1969). Pulsed nuclear magnetic resonance studies of porosity, movable fluid, and permeability of sandstones. J Pet Technol 21:775–786. <https://doi.org/10.2118/2045-PA>

[265] Tinni A, Odusina E, Sulucarnain I, Sondergeld C, Rai CS (2015). Nuclear-magnetic-resonance response of brine, oil, and methane in organic-rich shales. SPE Reserv Eval Eng 18:400–406. <https://doi.org/10.2118/168971-PA>

[266] Toumelin E, Torres-Verdin C, Chen S, Fischer DM (2002). Analysis of NMR diffusion coupling effects in two-phase carbonate rocks: comparison of measurements with Monte Carlo simulations. In: SPWLA 43rd Annual logging symposium.

[267] Turov VV, Mironyuk IF (1998). Adsorption layers of water on the surface of hydrophilic, hydrophobic and mixed silicas. Colloids Surfaces A Physicochem Eng Asp 134:257–263. <https://doi.org/10.1016/S0927-7757(97)00225-2>

[268] Turov VV, Leboda R, Bogillo VI, Skubiszewska-Ziȩba J (1997). Study of hydrated structures on the surface of mesoporous silicas and carbosils by 1H NMR spectroscopy of adsorbed water. Langmuir 13:1237–1244. <https://doi.org/10.1021/la951565p>

[269] Valori A, Hursan G (2017). Laboratory and downhole wettability from NMR T1/T2 ratio. Petrophysics 58:352–365.

[270] Valori A, Hursan G, Ma SM (2017). Laboratory and downhole wettability from NMR T1/T2 ratio. Petrophysics 58:352–365.

[271] Valori A, Nicot B (2019). A review of 60 years of NMR wettability. Petrophys– SPWLA J Form Eval Reserv Descr 60:255–263. <https://doi.org/10.30632/PJV60N2-2019a3>

[272] Valori A, Ali F, Abdallah W (2018). Downhole wettability: the potential of NMR. In: SPE EOR conference oil gas West Asia.

[273] Venkataramanan L, Song Y-Q, Hurlimann MD (2002). Solving Fredholm integrals of the first kind with tensor product structure in 2 and 2.5 dimensions. IEEE Trans Signal Process 50:1017–1026. <https://doi.org/10.1109/78.995059>

[274] Venkataramanan L, Hurlimann MD, Tarvin JA, Fellah K, Acero-Allard D, Seleznev NV (2014). Experimental study of the effects of wettability and fluid saturation on nuclear magnetic resonance and dielectric measurements in limestone. Petrophys SPWLA J Form Eval Reserv Descr 55:572–586.

[275] Vij J, Saraiya R, Saumya S, Sarkar SK, Majumdar C (2018). LWD as the absolute formation evaluation technology: present-day capabilities, limitations, and future developments of LWD technology. In: SPWLA 2nd Asia Pacific tech symp.

[276] Vold RL, Waugh JS, Klein MP, Phelps DE (1968). Measurement of spin relaxation in complex systems. J Chem Phys 48:3831–3832. <https://doi.org/10.1063/1.1669699>

[277] Wahba G, Wang Y (1990). When is the optimal regularization parameter insensitive to the choice of the loss function? Commun Stat - Theory Methods 19:1685–1700. <https://doi.org/10.1080/03610929008830285>

[278] Walstra P (1993). Principles of emulsion formation. Chem Eng Sci 48:333–349. <https://doi.org/10.1016/0009-2509(93)80021-H>

[279] Wang H, Alvarado V, McLaughlin JF, Bagdonas DA, Kaszuba JP, Campbell E, Grana D (2018a). Low-field nuclear magnetic resonance characterization of carbonate and sandstone reservoirs from rock spring uplift of wyoming. J Geophys Res Solid Earth 123:7444–7460. <https://doi.org/10.1029/2018JB015779>

[280] Wang J, Xiao L, Liao G, Zhang Y, Guo L, Arns CH, Sun Z (2018b). Theoretical investigation of heterogeneous wettability in porous media using NMR. Sci Rep 8:13450. <https://doi.org/10.1038/s41598-018-31803-w>

[281] Wang H, Alvarado V, Bagdonas DA, McLaughlin JF, Kaszuba JP, Grana D, Campbell E, Ng K (2021a). Effect of CO2-brine-rock reactions on pore architecture and permeability in dolostone: implications for CO2 storage and EOR. Int J Greenh Gas Control 107:103283. <https://doi.org/10.1016/j.ijggc.2021.103283>

[282] Wang H, Huang T, Granick S (2021b). Using NMR to test molecular mobility during a chemical reaction. J Phys Chem Lett 12:2370–2375. <https://doi.org/10.1021/acs.jpclett.1c00066>

[283] Wang Y, Medellin D, Torres-Verdín C (2019). Estimating capillary pressure from NMR measurements using a pore-size-dependent fluid substitution method. In: SPWLA 60th annual logging symposium transactions. Society of Petrophysicists and Well Log Analysts, pp. 1–9.

[284] Washburn KE (2014). Relaxation mechanisms and shales. Concepts Magn Reson Part A 43A:57–78. <https://doi.org/10.1002/cmr.a.21302>

[285] Washburn KE, Callaghan PT (2006). Tracking pore to pore exchange using relaxation exchange spectroscopy. Phys Rev Lett 97:175502. <https://doi.org/10.1103/PhysRevLett.97.175502>

[286] Washburn KE, Anderssen E, Vogt SJ, Seymour JD, Birdwell JE, Kirkland CM, Codd SL (2015). Simultaneous Gaussian and exponential inversion for improved analysis of shales by NMR relaxometry. J Magn Reson 250:7–16. <https://doi.org/10.1016/j.jmr.2014.10.015>

[287] Watson AT, Chang CTP (1997). Characterizing porous media with NMR methods. Prog Nucl Magn Reson Spectrosc 31:343–386. <https://doi.org/10.1016/S0079-6565(97)00053-8>

[288] Weber D, Mitchell J, McGregor J, Gladden LF (2009). Comparing strengths of surface interactions for reactants and solvents in porous catalysts using two-dimensional NMR relaxation correlations. J Phys Chem C 113:6610–6615. <https://doi.org/10.1021/jp811246j>

[289] Weil JA, Bolton JR (2006). Electron paramagnetic resonance, contemporary physics. Wiley, Hoboken.

[290] Wiemers-Meyer S, Winter M, Nowak S (2019). NMR as a powerful tool to study lithium ion battery electrolytes. Academic Press, London, pp 121–162.

[291] Willis SA, Stait-Gardner T, Torres AM, Price WS (2016). Fundamentals of diffusion measurements using NMR. In: Valiullin R (ed) Diffusion NMR of confined systems: fluid transport in porous solids and heterogeneous materials. The Royal Society of Chemistry, Cambridge, pp 16–51.

[292] Wilson JD (1992). Statistical approach to the solution of first-kind integral equations arising in the study of materials and their properties. J Mater Sci 27:3911–3924. <https://doi.org/10.1007/BF00545476>

[293] Wong KC (2014). Review of NMR spectroscopy: basic principles, concepts and applications in chemistry. J Chem Educ 91:1103–1104. <https://doi.org/10.1021/ed500324w>

[294] Wong SF, Lim JS, Dol SS (2015). Crude oil emulsion: a review on formation, classification and stability of water-in-oil emulsions. J Pet Sci Eng 135:498–504. <https://doi.org/10.1016/j.petrol.2015.10.006>

[295] Worden RH, Morad S (1999). Clay minerals in sandstones: controls on formation, distribution and evolution. In: Clay mineral cements in sandstones. Blackwell Publishing Ltd.: Oxford, UK, pp. 1–41.

[296] Wu J, Fan Y, Wu F, Li C (2019). Combining large-sized model flow experiment and NMR measurement to investigate drilling induced formation damage in sandstone reservoir. J Pet Sci Eng 176:85–96. <https://doi.org/10.1016/j.petrol.2019.01.005>

[297] Wu B, Xie R, Xu C, Wei H, Wang S, Liu J (2021). A new method for predicting capillary pressure curves based on NMR echo data: Sandstone as an example. J Pet Sci Eng 202:108581. <https://doi.org/10.1016/j.petrol.2021.108581>

[298] Xiao L, Mao Z, Zou C, Jin Y, Zhu J (2016). A new methodology of constructing pseudo capillary pressure (Pc) curves from nuclear magnetic resonance (NMR) logs. J Pet Sci Eng 147:154–167. <https://doi.org/10.1016/j.petrol.2016.05.015>

[299] Xie R, Xiao L, Wang Z, Dunn KJ (2008). The influence factors of NMR logging porosity in complex fluid reservoir. Sci China Ser D. Earth Sci 51(2):212–217.

[300] Yan W, Sun J, Sun Y, Golsanami N (2018). A robust NMR method to measure porosity of low porosity rocks. Microporous Mesoporous Mater 269:113–117. <https://doi.org/10.1016/j.micromeso.2018.02.022>

[301] Yang D, Kausik R (2016). 23Na and 1H NMR relaxometry of shale at high magnetic field. Energy Fuels 30:4509–4519. <https://doi.org/10.1021/acs.energyfuels.6b00130>

[302] Yang K, Li M, Ling NNA, May EF, Connolly PRJ, Esteban L, Clennell MB, Mahmoud M, El-Husseiny A, Adebayo AR, Elsayed MM, Johns ML (2019). Quantitative tortuosity measurements of carbonate rocks using pulsed field gradient NMR. Transp Porous Media 130:847–865. <https://doi.org/10.1007/s11242-019-01341-8>

[303] Yang K, Connolly PRJ, Li M, Seltzer SJ, McCarty DK, Mahmoud M, El-Husseiny A, May EF, Johns ML (2020). Shale rock core analysis using NMR: effect of bitumen and water content. J Pet Sci Eng 195:107847. <https://doi.org/10.1016/j.petrol.2020.107847>

[304] Yu Z, Zhang Y, Xiao L, Liao G (2019). Characterization of porous media by T2–T2 correlation beyond fast diffusion limit. Magn Reson Imaging 56:19–23. <https://doi.org/10.1016/j.mri.2018.10.008>

[305] Zaki NN, Carbonell RG, Kilpatrick PK (2003). A novel process for demulsification of water-in-crude oil emulsions by dense carbon dioxide. Ind Eng Chem Res 42:6661–6672. <https://doi.org/10.1021/ie0303597>

[306] Zhang GQ, Huang CC, Hirasaki GJ (2000). Interpretation of wettability in sandstones with NMR analysis. Log Anal 41:223–233.

[307] Zhang T, Ellis GS, Ruppel SC, Milliken K, Yang R (2012). Effect of organic-matter type and thermal maturity on methane adsorption in shale-gas systems. Org Geochem 47:120–131. <https://doi.org/10.1016/j.orggeochem.2012.03.012>

[308] Zhang Y, Xiao L, Liao G, Song YQ (2016). Direct correlation of diffusion and pore size distributions with low field NMR. J Magn Reson 269:196–202. <https://doi.org/10.1016/j.jmr.2016.06.013>

[309] Zhang G, Hirasaki G, House W (1998). Diffusion in internal field gradients. In: symposium soc core.

[310] Zhao P, Wang L, Xu C, Fu J, Shi Y, Mao Z, Xiao D (2020a). Nuclear magnetic resonance surface relaxivity and its advanced application in calculating pore size distributions. Mar Pet Geol 111:66–74. <https://doi.org/10.1016/j.marpetgeo.2019.08.002>

[311] Zhao Y, Zhang Y, Lei X, Zhang Yi, Song Y (2020b). CO2 flooding enhanced oil recovery evaluated using magnetic resonance imaging technique. Energy 203:117878. <https://doi.org/10.1016/j.energy.2020.117878>

[312] Zhu D-Y, Deng Z-H, Chen S-W (2021). A review of nuclear magnetic resonance (NMR) technology applied in the characterization of polymer gels for petroleum reservoir conformance control. Pet Sci. <https://doi.org/10.1016/j.petsci.2021.09.008>

[313] Zia K, Siddiqui T, Ali S, Farooq I, Zafar MS, Khurshid Z (2019). Nuclear magnetic resonance spectroscopy for medical and dental applications: a comprehensive review. Eur J Dent 13:124–128. <https://doi.org/10.1055/s-0039-1688654>
