# Nuclear Magnetic Resonance in the Oil and Gas Industry: Unveiling Subsurface Complexity Through Advanced Characterization and Emerging Technologies

## Contents

1. Introduction 2
2. Fundamentals of Nuclear Magnetic Resonance 6
    2.1 Relaxation Mechanisms 8
    2.2 Pulse Sequences 11
    2.3 Mathematical Background and Signal Processing 13
3. NMR Applications in Petrophysics 14
    3.1 Porosity Determination 15
    3.2 Pore Size Distribution 16
    3.3 Permeability Estimation 19
    3.4 Wettability and Fluid Typing 22
4. NMR in Enhanced Oil Recovery (EOR) Monitoring 24
    4.1 Chemical EOR Monitoring 26
    4.2 Gas Injection EOR Monitoring 27
    4.3 Thermal EOR Monitoring 28
    4.4 Techniques for Saturation and Fluid Distribution 28
    4.5 Advantages and Limitations of NMR in EOR 32
5. NMR in Unconventional Reservoirs 33
    5.1 Shale Characterization 33
    5.2 Tight Gas Sands 37
    5.3 Heavy Oil and Oil Sands 38
6. Advanced NMR Techniques and Future Trends 39
    6.1 Multi-Dimensional NMR 40
    6.2 High-Field NMR 43
    6.3 NMR Imaging (MRI) 45
    6.4 Future Directions in NMR Research 47
7. Conclusion 48
8. References 52

## Abstract

Nuclear Magnetic Resonance (NMR) has emerged as an indispensable, non-destructive analytical technique for characterizing porous media and fluid behavior across the oil and gas industry. This comprehensive survey reviews the fundamental principles of NMR, elucidating its diverse applications from laboratory core analysis to field-scale operations. We detail its pivotal role in petrophysical characterization, including the quantification of porosity, pore size distribution, permeability, and fluid saturation, and its unique ability to differentiate fluid phases and assess wettability. The paper highlights NMR's significant contributions to Enhanced Oil Recovery (EOR) monitoring, providing granular insights into fluid displacement and residual oil saturation, and its critical application in challenging unconventional reservoirs for detailed pore structure and fluid typing. Furthermore, its integration into Logging While Drilling (LWD) operations for real-time formation evaluation is discussed. While offering unparalleled insights, NMR applications in complex geological systems face challenges such as internal magnetic field gradients, pore coupling effects, and data interpretation complexities in low-porosity formations. Addressing these, we explore future directions in NMR research, including the optimization of pulse sequences, the integration of advanced computational methods like Machine Learning and Artificial Intelligence for data interpretation, the development of more efficient LWD hardware, and the necessity for multi-modal data integration and reservoir-representative measurement conditions. This review underscores NMR's continued evolution as a transformative tool, driving advancements in hydrocarbon exploration, development, and production optimization.

## 1. Introduction

Nuclear Magnetic Resonance (NMR), a highly significant and versatile analytical tool rooted in the fundamental quantum mechanical property of nuclear spin, offers unparalleled, non-destructive insights into the molecular structure, dynamics, and environment of materials [1, 2, 3]. By detecting the magnetic response of atomic nuclei when placed in a strong static magnetic field and perturbed by radiofrequency pulses, NMR provides unique information often inaccessible through other analytical methods, making it indispensable for both fundamental research and industrial applications. Its ability to probe the microscopic properties of matter, particularly the behavior of hydrogen-containing fluids, has cemented its role in diverse fields ranging from medical imaging, where it underpins Magnetic Resonance Imaging (MRI) [4], to materials science and chemical analysis [1, 4], and in the oil and gas industry [5, 6, 7, 8]. Figure 1 illustrates a fundamental aspect of NMR data acquisition and processing: the transformation of raw Carr-Purcell-Meiboom-Gill (CPMG) signals into T2 relaxation spectra [9, 10, 11, 12]. While this specific example is from a biological sample (sea cucumber), the underlying principles of acquiring decaying echo trains and inverting them to obtain a distribution of relaxation times are universally applicable across all NMR measurements, including those performed in the oil and gas industry for characterizing porous media and fluids.

<center>Figure 1: The CPMG signals (A) and T2 relaxation spectra (B) acquired from the raw data obtained for lightly dried and salted dried sea cucumber. Adapted from Hriberšek, Matej, 2024 [13]. </center>

In the realm of the oil and gas industry, NMR has emerged as a transformative technology, revolutionizing the way reservoirs are characterized and fluids are analyzed, both in laboratory settings and at the field scale. Its application in earth sciences historically dates back decades, with its principles being employed in proton-precession magnetometers for over 50 years and in borehole and core exploration tools for over 40 years. The development of early nuclear magnetism logging (NML) tools, which responded solely to formation fluids, marked a significant advancement. As shown in Figure 2, these tools provided continuous 'Free Fluid Index' (FFI) logs. The FFI represents the volume of mobile fluids (typically water and hydrocarbons) that can be freely produced from the formation, distinguishing them from irreducible bound fluids. Comparing the FFI curve with core-analysis porosity and conventional logs, as depicted, demonstrated the early utility of NML in directly identifying fluid-bearing zones and estimating effective porosity, offering insights into permeability and wettability that were less directly obtainable from other logs. The inherent relationship between the NMR signal and the hydrogen nuclei present in formation fluids (oil, water, and gas) allows for direct measurements of critical petrophysical properties that are crucial for effective hydrocarbon exploration, production, and reservoir management. Unlike conventional logging tools that often rely on indirect measurements influenced by lithology, NMR offers direct information about the pore structure and fluid properties, independent of the rock matrix itself. This distinct advantage enables a more accurate and comprehensive understanding of complex porous media, which is paramount for optimizing recovery strategies and assessing reservoir potential.

The profound importance of NMR in the oil and gas sector necessitates a thorough understanding of its fundamental principles and diverse applications. Accurate determination of petrophysical properties such as porosity, pore size distribution, fluid saturation, permeability, capillary pressure, and wettability is vital for evaluating oil and gas reservoirs and precisely estimating reserves and potential recovery approaches. Figure 3 exemplifies how NMR data can be used to derive pore size distributions, often in conjunction with Mercury Intrusion Porosimetry (MICP). By correlating the NMR T2 relaxation time distribution, which is sensitive to pore geometry, with the direct pore throat size measurements from MICP on a subset of samples, a robust relationship can be established. This enables the non-destructive and efficient determination of pore size distribution for other samples from the same formation, providing crucial insights into the reservoir's storage capacity and fluid flow characteristics. Low-field NMR, in particular, has proven its robustness by offering capabilities for both field-scale logging and detailed laboratory-scale core analysis. This dual capability allows for reliable cross-validation between core data and logging measurements, enhancing the overall accuracy of reservoir characterization. Furthermore, NMR's sensitivity to rock-fluid interactions makes it an invaluable tool for evaluating Enhanced Oil Recovery (EOR) techniques, providing a more detailed assessment of fluid recovery from different pore systems compared to macroscopic techniques that only yield bulk recovery data. The continuous advancements in NMR technology, including the development of advanced pulse sequences and improved hardware for logging while drilling (LWD) operations, continue to expand its utility and impact within the industry. A comprehensive grasp of these aspects is essential for leveraging NMR to its full potential in tackling the complex challenges associated with modern hydrocarbon exploration and production, especially in increasingly complex and unconventional reservoirs.

<center>Figure 2: Comparison of Free Fluid Index (FFI) Curve To Core-Analysis Porosity and to conventional Logs. Adapted from Brown, R.J.S., et al., 1960 [14]. </center>

<center>Figure 3: Pore size distribution from NMR and MICP from Pierre Shale: The MICP curve (solid black line) can be used to shift the original NMR T2 time relaxation distribution (solid blue curve), into alignment with a quantitative measure of pore size. The time abscissa axis from the original NMR T2 distribution becomes a pore throat size abscissa axis by this shifting method. Adapted from Josh, M., et al., 2012 [15]. </center>

This survey aims to provide a comprehensive overview of Nuclear Magnetic Resonance technology and its extensive applications within the oil and gas industry. The subsequent sections will delve into various facets of NMR, building from its foundational principles to its most advanced applications. Section 2 will establish the theoretical groundwork, explaining the core physics governing NMR phenomena and the different relaxation mechanisms. Following this, Section 3 will explore the diverse applications of NMR in petrophysical characterization, detailing how it is used to determine key reservoir properties such as porosity, pore size distribution, permeability, fluid saturation, and wettability. Section 4 will then focus on NMR's crucial role in monitoring and evaluating Enhanced Oil Recovery (EOR) processes, highlighting its ability to track fluid displacement and assess recovery efficiency. The application of NMR in the challenging domain of unconventional reservoirs, including shale oil and gas, will be discussed in Section 5, emphasizing its unique capabilities in characterizing tight formations. Finally, Section 6 will examine advanced NMR techniques, such as multi-dimensional NMR and magnetic resonance imaging, and discuss future research directions and challenges, while Section 7 will provide concluding remarks.

## 2. Fundamentals of Nuclear Magnetic Resonance

Nuclear Magnetic Resonance (NMR) stands as a powerful analytical technique, fundamentally rooted in the quantum mechanical properties of atomic nuclei. The core principle of NMR hinges on the inherent magnetism of certain nuclei, which possess both a nuclear spin angular momentum and a magnetic moment. These properties are quantified by two critical constants: the nuclear spin quantum number, *J*, and the gyromagnetic ratio, *γ*. A nucleus is considered "NMR active" if its nuclear spin quantum number, *J*, is greater than zero. While a high gyromagnetic ratio, *γ*, is beneficial for achieving a strong detectable signal, the fundamental criterion for NMR activity is the non-zero nuclear spin. In the context of oil and gas applications, hydrogen nuclei, or protons ($^1$H), are predominantly the species of interest due to their high natural abundance in formation fluids and favorable NMR properties [5, 14, 7, 6]. Other common elements found in geological formations, such as carbon, oxygen, magnesium, silicon, sulfur, and calcium, largely consist of isotopes that lack a magnetic moment or spin, rendering them NMR inactive. Even elements like potassium, iron, sodium, and aluminum, which may possess weak magnetic moments, typically offer low detection efficiency in typical NMR logging scenarios. Consequently, the NMR signal in these applications primarily originates from hydrogen nuclei present in the fluid phase within porous media.

<center>Figure 4: Schematic of the NMR apparatus used for the NMR test. Adapted from Gao, Hui, et al., 2015 [16]. </center>

The fundamental concept of an NMR experiment involves placing a sample containing these NMR-active nuclei within a strong, static external magnetic field, conventionally denoted as $B_z$ [4, 3]. In this magnetic field, the nuclear spins align either parallel or anti-parallel to the field, creating a net macroscopic magnetization. This net magnetization, proportional to the total number of spins within the sampled volume, is in thermal equilibrium with the static field. The nuclei also undergo a precessional motion around the $B_z$ field, akin to a spinning top in a gravitational field. The frequency of this precession, known as the Larmor frequency ($f$), is directly proportional to the strength of the static magnetic field and the gyromagnetic ratio of the nucleus, as described by the Larmor equation:

$$f = \frac{\gamma}{2\pi} B_z \tag{1}$$

where $f$ is the Larmor frequency (typically in MHz), $\gamma$ is the gyromagnetic ratio of the nucleus (in MHz/T), and $B_z$ is the strength of the static magnetic field (in T). To induce a detectable NMR signal, a second, oscillating magnetic field, $B_1$, in the form of radio-frequency (RF) pulses, is applied perpendicular to the static $B_z$ field. This $B_1$ field, when applied at the Larmor frequency, excites the nuclear spins, tipping the net magnetization away from its equilibrium alignment into the transverse ($x - y$) plane. Upon removal of the RF pulse, the excited spins begin to precess coherently in the transverse plane, generating a detectable oscillating signal. This process marks the initiation of the relaxation phenomenon, where the nuclear spins return to their equilibrium state, and the decay of this oscillating signal provides crucial information about molecular dynamics, adsorption, confinement, and pore characteristics within porous media [6, 17, 18, 19].

### 2.1 Relaxation Mechanisms

The return of the nuclear magnetization to its equilibrium state after excitation is characterized by two distinct relaxation processes: longitudinal relaxation ($T_1$) and transverse relaxation ($T_2$). These relaxation times are highly sensitive probes of the molecular environment and interactions within a sample.

Longitudinal relaxation, also known as spin-lattice relaxation, describes the exponential recovery of the magnetization component parallel to the static magnetic field ($B_z$) back to its thermal equilibrium value. This process involves the exchange of energy between the spin system and its surrounding thermal environment, or "lattice," which includes the molecular motion of the fluid and the solid matrix. The recovery of longitudinal magnetization, $M_z$, over time, $t_I$, can be expressed as:

$$\frac{M_z(t_I)}{M_0} = 1 - 2e^{-t_I / T_1} \tag{2}$$

where $M_0$ is the equilibrium magnetization. The $T_1$ relaxation is influenced by the statistical likelihood of magnetic interactions between a proton and other nearby magnetic entities, including the pore walls. For fluids confined within porous media, $T_1$ values are typically shorter in smaller pores and longer in larger pores, reflecting the extent of interaction with the pore surfaces [20, 17, 21, 22].

Transverse relaxation, or spin-spin relaxation, describes the irreversible decay of the magnetization in the transverse ($x - y$) plane due to the dephasing of individual nuclear spins [23, 24, 25]. This dephasing arises from spin-spin interactions and local magnetic field inhomogeneities experienced by each nucleus. The decay of transverse magnetization, $M_{xy}$, can be described by:

$$\frac{M_{xy}(t)}{M_0} = e^{-t / T_2} \tag{3}$$

In homogeneous bulk liquids, $T_1$ and $T_2$ are often similar. However, in porous media, $T_2$ is invariably shorter than $T_1$ because it is additionally affected by static magnetic field gradients inherent in the sample, which cause faster dephasing [26, 27, 28, 29].

Several factors intricately affect both longitudinal and transverse relaxation times. Molecular motion and fluid viscosity play a significant role, as described by the Bloembergen-Purcell-Pound (BPP) theory [23, 30, 31]. Generally, higher viscosity or reduced molecular mobility leads to shorter relaxation times [32, 33, 30]. When liquid molecules adsorb onto a solid surface, their mobility is restricted, consequently enhancing relaxation and leading to reduced $T_1$ and $T_2$ values [18, 34, 35, 36].

Surface interactions are particularly critical in porous media. NMR relaxation in such systems is often modeled by a biphasic fast-exchange mechanism, where fluid molecules rapidly exchange between a surface-adsorbed layer and the bulk fluid within the pore [20, 31, 30, 17]. The observed relaxation rate ($1 / T_{obs}$) is a weighted average of the bulk fluid relaxation rate ($1 / T_{bulk}$) and the surface relaxation rate ($1 / T_{surface}$):

$$\frac{1}{T_{obs}} = P\frac{1}{T_{surface}} +(1 - P)\frac{1}{T_{bulk}} \tag{4}$$

where $P$ represents the population fraction of the adsorbed surface layer. This population can be related to the pore structure's surface-to-volume ratio ($S / V$) by $P = \delta (S / V)$, where $\delta$ is the characteristic thickness of the adsorbed layer. This general form accounts for contributions from both surface and bulk fluid environments. For situations where fluid molecules diffuse rapidly between the surface and bulk regions within a pore (known as the "fast diffusion limit") [37, 17], and the surface relaxation rate significantly dominates over the bulk fluid relaxation rate, Equation (4) simplifies to:

$$\frac{1}{T_{obs}}\approx \rho_s\frac{S}{V} \tag{5}$$

Here, $\rho_s$ is the surface relaxivity, a crucial parameter that quantifies the ability of the pore wall to enhance relaxation and provides a direct link between the observed relaxation rate and pore size [38, 21, 39, 12]. Thus, in water-saturated rocks, shorter $T_1$ and $T_2$ values indicate smaller pores due to the increased $S/V$ ratio [40, 41, 42, 22].

Paramagnetic substances and internal magnetic field gradients also profoundly influence relaxation [26, 43, 29, 44]. Paramagnetic ions, often present on pore surfaces, possess unpaired electrons with very large magnetic moments (the electron gyromagnetic ratio is approximately 650 times that of a proton). These strong local magnetic fields significantly accelerate relaxation, especially $T_2$ [43, 45, 46, 44]. Furthermore, magnetic susceptibility differences between the solid rock matrix and the fluid phases induce internal magnetic field gradients within the pores. These gradients cause additional dephasing of nuclear spins, leading to a further reduction in $T_2$ (but not $T_1$) [26, 27, 29, 47]. The effective transverse relaxation time, $T_2^E$, in the presence of such inhomogeneities, is given by:

$$\frac{1}{T_2^E} = \frac{1}{T_2} +\frac{1}{12}\gamma^2 G^2 Dt_e^2 \tag{6}$$

Here, the $1/T_2$ term in the denominator represents the transverse relaxation rate in the absence of internal magnetic field gradients, encompassing contributions from both bulk fluid relaxation and surface relaxation [48, 29, 28, 49]. Furthermore, $G$ is the internal magnetic field gradient, $D$ is the diffusion coefficient, and $t_e$ is the echo time. The effects of static field inhomogeneities can be mitigated through specific pulse sequences, enabling the measurement of the true $T_2$ [48, 27, 50, 29]. The ratio $T_1 / T_2$ can also provide valuable insights into pore structure, mineralogy, and even the strength of surface adsorption interactions [18, 43, 51, 36].

### 2.2 Pulse Sequences

The manipulation and detection of nuclear magnetization in NMR experiments are achieved through precisely timed sequences of RF pulses, known as pulse sequences. These sequences are designed to selectively excite, evolve, and detect the NMR signal, allowing the extraction of specific physical parameters.

The most fundamental detection method is the Free Induction Decay (FID) [14, 52, 53], which involves applying a 90-degree RF pulse to tip the magnetization into the transverse plane, followed by the direct observation of the decaying signal as the spins dephase. While simple, FID is highly sensitive to magnetic field inhomogeneities [26, 54, 27, 55].

To measure $T_1$ relaxation, the Inversion Recovery (IR) pulse sequence is commonly employed [56, 22, 12, 57]. This sequence begins with a 180-degree RF pulse that inverts the equilibrium longitudinal magnetization. The system is then allowed to recover for a variable inversion time, $t_I$. Following this recovery period, a 90-degree RF pulse is applied to flip the recovering longitudinal magnetization into the transverse plane, where the signal amplitude is detected. By varying $t_I$ and measuring the corresponding signal amplitude, the exponential recovery curve can be fitted to determine the $T_1$ relaxation time.

For $T_2$ measurements, the Carr-Purcell-Meiboom-Gill (CPMG) pulse sequence is the standard [58, 59, 60]. This sequence effectively mitigates the dephasing effects caused by static magnetic field inhomogeneities [58]. The CPMG sequence starts with a 90-degree RF pulse that tips the magnetization into the transverse plane. This is followed by a series of 180-degree refocusing pulses, each separated by an echo time ($t_e$). Each 180-degree pulse inverts the phases of the spins, causing them to rephase and form spin echoes at times $2t_e$, $4t_e$, and so on, after the initial 90-degree pulse [53, 58]. The amplitude of these spin echoes decays exponentially due to true $T_2$ relaxation and other irreversible dephasing mechanisms [9, 27]. The CPMG sequence allows for the accurate determination of $T_2$ by fitting the envelope of these echo amplitudes [61, 9].

$$M_{xy}(i\cdot t_e) = M_0e^{-(i\cdot t_e) / T_2} \tag{7}$$

where $i$ is the echo number.

Beyond these basic sequences, advanced two-dimensional (2D) NMR techniques offer more comprehensive insights by correlating different NMR parameters [62, 57, 63]. These methods typically involve encoding one parameter in an indirect dimension and detecting another in a direct dimension. For instance, $T_1 - T_2$ correlation, achieved by combining an IR component followed by a CPMG train, provides a 2D probability density map showing combinations of $T_1$ and $T_2$ values [64, 11, 65]. This map is particularly useful for differentiating fluid types and pore sizes [62, 66, 50, 12]. The acquired signal for $T_1 - T_2$ correlation can be expressed as:

$$M(t_1,t_e) = \sum_{j=1}^{N_1}\sum_{k=1}^{N_2}F(T_{1j},T_{2k})(1 - 2e^{-t_1 / T_{1j}})e^{-t_e / T_{2k}} \tag{8}$$

where $F(T_{1j},T_{2k})$ is the 2D distribution function.

Another powerful 2D technique is $T_2 - D$ correlation, which maps transverse relaxation time against the diffusion coefficient. This is typically achieved by combining a Pulsed Field Gradient Spin Echo (PGSE) or Adiabatic Pulsed Gradient Spin Echo (APGSTE) sequence for diffusion encoding with a CPMG sequence for $T_2$ detection. The $T_2 - D$ maps are invaluable for robustly differentiating oil, gas, and water signals in complex fluid mixtures. The signal acquired from such a sequence is given by:

$$S(g,t_e) = \sum_{j=1}^{N_1}\sum_{k=1}^{N_2}F(D_j,T_{2k})e^{-D_j\gamma^{2}g^{2}\delta^{2}(\Delta -\delta/3)}e^{-t_e / T_{2k}} \tag{9}$$

where g is the magnetic field gradient strength, δ is the duration of the gradient pulse, and ∆ is the diffusion time.

Finally, T2−T2 exchange spectroscopy, achieved by two CPMG echo trains separated by a mixing time (∆mixing), is used to study diffusive and chemical exchange processes between different pore environments. The appearance of off-diagonal peaks in the 2D map signifies exchange, with their intensity changes over varying ∆mixing providing information on the rate and extent of this exchange. The acquired 2D data for this sequence is given by:

$$S(t_e^{(1)}, t_e^{(2)}, \Delta_{mixing}) = \sum_{j=1}^{N_1}\sum_{k=1}^{N_2}F(T_{2j}^{(1)}, T_{2k}^{(2)})e^{-t_e^{(1)}/T_{2j}^{(1)}}e^{-t_e^{(2)}/T_{2k}^{(2)}} \tag{10}$$

These advanced sequences, particularly 2D NMR, are computationally intensive but, with techniques like Singular Value Decomposition (SVD) and kernel separability, have become tractable on standard computing platforms.

### 2.3 Mathematical Background and Signal Processing

The raw NMR signal acquired in the time domain represents the decay or recovery of magnetization. To extract meaningful physical parameters such as relaxation time distributions (spectra), these time-domain signals must be transformed and processed. The underlying mathematical framework for this transformation is often based on the Fredholm integral equation of the first kind:

$$\phi(t) = \int K(t, T)F(T)dT + \epsilon(t) \tag{11}$$

Here, $\phi(t)$ represents the experimentally acquired signal, $F(T)$ is the desired distribution of relaxation times (e.g., T1 or T2), $K(t, T)$ is the kernel function describing the expected exponential decay or growth based on the specific pulse sequence, and $\epsilon(t)$ accounts for experimental noise. For T1 measurements, the kernel typically represents an exponential growth, while for T2 it represents an exponential decay.

To solve for the distribution $F(T)$, the integral equation is typically discretized and rewritten in a vector-matrix form:

$$\mathbf{M} = \mathbf{KF} + \boldsymbol{\epsilon} \tag{12}$$

where $\mathbf{M}$ is the acquired data vector, $\mathbf{K}$ is the kernel matrix, $\mathbf{F}$ is the target probability distribution vector, and $\boldsymbol{\epsilon}$ is the noise vector. The goal is to find $\mathbf{F}$ that minimizes the difference $\|\mathbf{M} - \mathbf{KF}\|^2$. However, this problem is inherently "ill-posed" in the presence of noise, meaning that small variations in the input data can lead to large, unphysical oscillations in the solution, and an infinite number of solutions may exist.

To obtain a stable and physically meaningful solution, regularization techniques are applied. Tikhonov regularization is a widely used method that adds a penalty term to the minimization function, promoting smoothness in the solution:

$$\min \|\mathbf{M} - \mathbf{KF}\|^2 + \alpha \|\mathbf{F}\|^2 \tag{13}$$

The parameter $\alpha$, known as the smoothing parameter, controls the trade-off between fitting the data faithfully and ensuring the smoothness of the solution. Its optimal value is often determined using robust algorithms like generalized cross-validation (GCV). Additionally, physical constraints, such as non-negativity of the distribution $\mathbf{F}(T)$ and defining a sensible range for relaxation times (e.g., $10^{-4}$ to $10^{4}$ seconds), are incorporated into the inversion process.

For 2D NMR data, the computational complexity significantly increases, as the kernel matrix can become exceptionally large. To overcome this, data compression techniques like Singular Value Decomposition (SVD) are employed to reduce the dimensionality of the problem by identifying and discarding less significant data components. Furthermore, the concept of kernel separability ($K = K_1 \otimes K_2$) is leveraged for 2D measurements, where the relaxation characteristics (e.g., $T_1$ and $T_2$) occur at separable time scales. This allows SVD to be performed on the individual components, making the 2D inversion problem manageable on standard computing systems.

Beyond inversion, general data processing techniques are crucial for improving the quality of NMR measurements. Enhancing the signal-to-noise ratio (SNR) is paramount for accurate measurements. This is often achieved by repeating the measurement cycle multiple times and stacking (averaging) the acquired signals, which causes the random noise to average out while the coherent NMR signal accumulates. These sophisticated mathematical and signal processing approaches are vital for transforming raw NMR data into interpretable distributions, enabling a deeper understanding of fluid properties and porous media characteristics.

## 3. NMR Applications in Petrophysics

Nuclear Magnetic Resonance (NMR) technology has emerged as an indispensable tool in the field of petrophysics, offering non-invasive and comprehensive characterization of reservoir rocks and their contained fluids [67, 6, 68, 69]. This section explores the fundamental applications of NMR in determining critical petrophysical properties, including porosity [70, 71, 72, 73], pore size distribution [22, 21, 12, 74], permeability [70, 7, 8, 75], and the assessment of wettability [76, 35, 77, 78] and fluid types [79, 80, 81, 82]. The insights derived from NMR measurements are crucial for effective reservoir evaluation, modeling, and optimizing hydrocarbon recovery strategies [67, 7, 8, 83].

### 3.1 Porosity Determination

Porosity, defined as the fraction of void space within a rock, represents the primary storage capacity of a reservoir and is a fundamental petrophysical property. NMR measurements are highly effective for quantifying porosity because the technique primarily detects hydrogen nuclei present in the fluids saturating the rock pores. When a rock sample is fully saturated with a single fluid, such as water, the magnitude of the detected NMR signal is directly proportional to the total pore volume of the rock, thereby providing a direct measure of its porosity. The accuracy and reliability of NMR porosity measurements are well-established, as demonstrated by comparisons with conventional methods. For instance, Figure 5 illustrates the strong agreement between NMR-derived porosity values and shipboard moisture and density (MAD) measurements across various sites, validating NMR as a precise technique for quantifying the total pore volume.

<center>Figure 5: Comparison of NMR porosity values with shipboard moisture and density (MAD) measurements for Sites C0011, C0012, and C0018. The NMR measurements match the MAD data very well. Adapted from Daigle, Hugh, et al., 2014 [84]. </center>

In heterogeneous carbonate reservoirs, which often exhibit a wide range of pore sizes from sub-micron to centimeter-scale vugs (Hidajat et al. 2004), NMR can effectively capture the total pore volume. However, the accuracy of NMR total porosity measurements is influenced by several experimental and fluid-specific factors. These include the echo spacing ($\tau_e$), which is the time between successive 180-degree radiofrequency pulses in a Carr-Purcell-Meiboom-Gill (CPMG) sequence; the strength of the static magnetic field ($B_0$); the hydrogen index (HI) of the fluid within the pores, which quantifies the hydrogen density relative to pure water; the repetition time (RT) between successive pulse sequences; and the rock temperature. Careful consideration and calibration of these parameters are essential to ensure the precision of porosity measurements.

When multiple fluid phases, such as oil, water, and gas, coexist within the pore space, NMR can differentiate and quantify the saturation of each phase [50, 12, 35, 66]. This is achieved by leveraging the distinct NMR signatures of different fluids, which often vary in their relaxation times and diffusion coefficients [35, 62, 85, 82]. The initial amplitude of the recorded echo signal is indicative of the total amount of hydrogen-containing fluids in the sample, from which the overall fluid saturation and total porosity can be derived [20, 7, 6, 86]. For instance, in an air-water system, as water-filled pores are emptied and replaced by air (which contains no hydrogen nuclei), the NMR signal amplitude decreases, reflecting the loss of hydrogen-containing fluid. This sensitivity to fluid content makes NMR an invaluable tool for understanding fluid distribution and saturation profiles within complex pore systems [12, 50, 87, 67].

### 3.2 Pore Size Distribution

One of the most significant applications of NMR in petrophysics is its ability to provide detailed information about the pore size distribution (PSD) within porous media. The transverse relaxation time ($T_2$) of fluids confined within pores is intimately related to the pore geometry, specifically the surface-to-volume ratio ($S/V$) of the pores. Smaller pores, with their higher $S/V$ ratios, lead to faster surface relaxation and thus shorter $T_2$ relaxation times, while larger pores correspond to longer $T_2$ times. This fundamental relationship allows the conversion of measured $T_2$ relaxation time distributions into pore size distributions. This relationship is conceptually illustrated in Figure 6, which depicts how shorter $T_2$ times correspond to smaller pores (e.g., micropores or clay-bound water), while longer $T_2$ times are indicative of larger pores (e.g., macropores or free fluids).

<center>Figure 6: Conceptual illustration of NMR $T_2$ relaxation time distribution and its relationship to pore size. Shorter $T_2$ times correspond to smaller pores (e.g., micropores or clay-bound water), while longer $T_2$ times are indicative of larger pores (e.g., macropores or free fluids). </center>

The general equation governing the transverse relaxation rate ($1/T_2$) in porous media is given by:

$$\frac{1}{T_2} = \frac{1}{T_{2,\mathrm{bulk}}} + \rho_2\left(\frac{S}{V}\right) + \frac{D\gamma^2 G^2 \tau_e^2}{12} \tag{14}$$

Here, $T_{2,\mathrm{bulk}}$ represents the bulk fluid relaxation time, $\rho_2$ is the surface relaxivity constant for transverse relaxation, $S/V$ is the surface-to-volume ratio of the pores, $D$ is the confined fluid diffusion coefficient, $\gamma$ is the gyromagnetic ratio of the proton nuclei, $G$ is the internal magnetic field gradient, and $\tau_e$ is the echo time. In many practical scenarios, particularly at low magnetic fields and short echo times, the bulk relaxation component (often very long, typically greater than 3000 ms) and the diffusion relaxation component (which can be minimized by reducing $\tau_e$ and performing experiments at low magnetic fields) become negligible [34, 27, 29, 26]. In such cases, the relaxation rate is dominated by surface interactions, simplifying the equation to $1/T_2 \approx \rho_2 (S/V)$. For idealized spherical pores, $S/V = 3/r$, where $r$ is the pore radius. This leads to a direct proportionality between $T_2$ and pore radius: $T_2 = C \cdot r$, where $C = 1/(3\rho_2)$ is a conversion factor [17, 21, 12, 40].

Despite its utility, challenges exist in accurately deriving PSD from NMR $T_2$ distributions. One significant issue is pore diffusive coupling, especially prevalent in pores smaller than 1 $\mu$m or in multi-modal pore systems like carbonates with distinct macro- and micropores (Fleury and Romero-Sarmiento, 2016)[88, 89, 90, 42]. In such cases, fluid molecules can diffuse between pores of different sizes during the relaxation measurement, leading to a single average relaxation time that does not accurately reflect the true PSD. Another major hurdle is the insufficient knowledge of the rock's surface relaxivity ($\rho_2$). While it is often assumed to be constant for a given sample, research indicates that $\rho_2$ can vary within a sample (Arns et al. 2006; Zhao et al. 2020a)[37, 39, 26], impacting the accuracy of the $T_2$-to-pore-size conversion. Extensive efforts, including the integration of micro-computed tomography ($\mu$CT) imaging with NMR, have been made to better evaluate surface relaxivity and achieve accurate overlap between Mercury Intrusion Porosimetry (MIP) and NMR relaxation distributions (Benavides et al. 2020; Connolly et al. 2019; Luo et al. 2015)[91, 92, 93, 94].

The $T_2$ spectrum itself can be decomposed into multiple independent component spectra, typically ranging from two to five, by fitting the distribution with Gaussian functions on a logarithmic $T_2$ axis [95, 41, 72, 96]. Each component spectrum can be assigned petrophysical significance, representing different pore types and fluid states [6, 93]. Common components include clay-bound water, capillary-bound fluid, and fluids within micropores and macropores [97, 98, 93, 42]. This decomposition provides a more granular understanding of the complex pore structures, which is particularly beneficial for characterizing tight sandstones and other unconventional reservoirs where pore connectivity and fluid mobility are critical [93, 99, 100, 101].

### 3.3 Permeability Estimation

Permeability, representing the ease with which fluids flow through a porous medium, is a crucial property for reservoir performance prediction. The basic principle of NMR application in rock permeability determination is the relationship that exists between NMR relaxation times and the pore geometry. While NMR does not directly measure fluid flow, it provides static petrophysical properties such as porosity and pore geometry, which can be empirically linked to permeability (Yang et al. 2019). This indirect estimation makes NMR a valuable tool for determining permeability both in laboratory core analysis and through downhole logging tools, enabling in-situ formation permeability estimation.

Several empirical correlations and models have been developed to estimate permeability from NMR measurements [6, 75, 71, 102]. One of the earliest approaches was presented by Seevers (1966), who combined the Kozeny equation for permeability with NMR relaxation times [20, 70]. Subsequent research by Banavar and Schwartz (1987), and Kenyon et al. (1988), further established relationships between permeability, NMR relaxation times, and porosity, often expressed in the general form [103, 104, 56, 105]:

$$k = a\phi^{b} T_{2, \text{gm}}^{c} \tag{15}$$

where $k$ is permeability, $\phi$ is porosity, $T_{2,\text{gm}}$ is the geometric mean of the $T_2$ distribution, and $a$, $b$, and $c$ are empirically derived constants.

Two widely recognized NMR-based permeability models are the Timur-Coates (T-C) model [75] and the Schlumberger Doll Research (SDR) model [104, 105]. The Timur model (Timur 1969), building upon the principle that all pores contribute to fluid transport based on their surface-to-volume ratio [20, 22, 21], is commonly expressed as:

$$k = C_T \phi^{4} (T_{2,\text{gm}})^{2} \tag{16}$$

where $C_T$ is an empirical constant and $T_{2,\text{gm}}$ is the geometric mean of the $T_2$ distribution.

The Coates et al. model (Coates et al., 1991 [75, 101]), often referred to as the Free Fluid Index (FFI) model, relates permeability to the ratio of movable fluids to irreducible fluids. It is formulated as:

$$k = C_C \phi^{4} \left(\frac{FFI}{BVI}\right)^{2} \tag{17}$$

where $C_C$ is an empirical parameter, FFI is the free fluid index (bulk volume movable, BVM), and BVI is the bulk volume irreducible (non-producible fluids) [20, 98, 106, 72]. The ratio $FFI/BVI$ serves as a measure of the specific internal pore surface [20, 31, 12].

Another frequently employed model is the SDR model:

$$k = C_{SDR} \phi^{4} T_{2,\text{gm}}^{2} \tag{18}$$

where $C_{SDR}$ is an empirical constant. Note that while the original SDR model often uses $\phi^4$, variations exist. The provided text uses $\phi^2 T_{2,\text{gm}}^2$, but later cites the SDR model as $\phi^4 T_{2,\text{gm}}^2$. For consistency with the equations in the text, the form $k = C_{SDR} \phi^{2} T_{2,\text{gm}}^{2}$ is used here as it appears in Equation (18) of the source. However, the more common SDR model is $k = C_{SDR} \phi^{4} T_{2,\text{gm}}^{2}$.

The practical application and comparison of these models are critical for accurate permeability prediction. For example, Figure 7 presents a comparison of NMR-derived hydraulic conductivity calculated using both the SDR and Timur-Coates equations. It highlights how these models, when calibrated with optimized empirical constants, can provide reliable estimates of hydraulic conductivity, which are essential for reservoir flow simulations and production forecasting. Both the T-C and SDR equations require empirically determined constants, which are typically calibrated through laboratory studies on consolidated materials to yield reliable permeability estimates in reservoir settings.

<center>Figure 7: (a) NMR-derived hydraulic conductivity calculated using the SDR and T-C equations with the standard empirical constants and a cutoff time of 33 ms. (b) NMR-derived hydraulic conductivity calculated using the SDR and T-C equations with the optimized empirical constants and a cutoff time of 33 ms. Adapted from Dlubac, Katherine, et al., 2013 [107]. </center>

Despite the widespread use of these models, several factors can affect the accuracy of NMR-derived permeability. Most correlations assume a constant surface relaxivity for rock samples, which may not hold true across heterogeneous formations, necessitating corrections for variations in $\rho_2$. Furthermore, these empirical models are primarily based on porosity-permeability relationships, which can be misleading in cases where high porosity does not translate to high permeability due to poor pore connectivity or small pore throats. Wettability of the porous medium also significantly influences permeability estimation (Ji et al. 2020; Elsayed et al. 2021a). To enhance the accuracy of NMR-derived permeability, it is crucial to calibrate the model parameters ($a, b, c, C_T, C_C, C_{SDR}$) to local reservoir data and to consider the NMR $T_2$ cutoff values that differentiate between movable and irreducible fluids, rather than relying solely on the bulk $T_2$ distribution.

### 3.4 Wettability and Fluid Typing

The wetting state of a reservoir rock, which describes the preference of the rock surface for one fluid over another, is a critical parameter influencing fluid distribution, flow dynamics, and ultimately, oil recovery efficiency. NMR offers robust capabilities for determining wettability indices and identifying fluid types (oil, water, and gas) within porous media, which is invaluable for characterizing mixed-wet and oil-wet reservoirs.

The interpretation of NMR $T_2$ distributions is pivotal for understanding the wetting state, based on the assumption of a relationship between pore throat and pore body sizes. This interpretation requires a thorough understanding of the fluids and rock properties involved. NMR relaxation times ($T_1$ and $T_2$) are sensitive not only to the size of the pores but also to the types of fluids present and the mineralogy of the pore walls (Kleinberg et al. 1994). This sensitivity allows NMR to differentiate between various fluid phases and their distribution within the pore network.

For fluid typing, the $T_2$ spectrum is particularly informative. As previously noted in the Porosity Determination subsection, the initial amplitude of the recorded echo signal directly indicates the total amount of hydrogen-containing fluids in a rock sample (Isah et al. 2021a, 2021b), enabling the calculation of overall fluid saturation and porosity. When a rock is fully saturated, the $T_2$ spectrum typically exhibits its highest amplitude, with larger pores corresponding to longer relaxation times. In contrast, during desaturation processes, such as air-water displacement, the amplitude of the longer $T_2$ components decreases as hydrogen-containing fluids are replaced by air (which yields no NMR signal) (Coates et al. 1997; Howard et al. 1995; Toumelin et al. 2002). This direct correlation between signal amplitude, relaxation time, and fluid content enables the precise determination of fluid saturation.

Advanced NMR techniques, such as two-dimensional (2D) NMR maps, significantly enhance the ability to characterize fluid distribution and proton mobility. For instance, Figure 8 demonstrates the utility of 2D NMR mappings in characterizing shales of different maturities, showcasing how these advanced techniques can reveal subtle differences in pore structure and fluid behavior that are critical for understanding reservoir quality in unconventional plays.

<center>Figure 8: Comparison of NMR 2-D mappings of shales in low-medium maturity (a) and high maturity (b). Adapted from Zhang, Jing-Ya, et al., 2023 [108]. </center>

$T_1 - T_2$ maps, for example, correlate longitudinal and transverse relaxation times, providing unique signatures for different fluids and their interactions with the pore surface (Habina et al. 2017; Tinni et al. 2015). The signal from these maps corresponds to hydrogen nuclei from the movable liquid fraction, making them useful for investigating both conventional and unconventional rocks. Similarly, $T_2$ - Diffusion ($T_2 - D$) maps improve fluid separation compared to $T_2$ relaxation alone, leading to more accurate calculations of effective surface relaxivities of fluids (Flaum et al. 2005; Liang et al. 2019; Minh et al. 2015).

The $T_1/T_2$ ratio is widely used for qualitatively assessing wettability (Katika et al. 2017; Valori et al. 2017; J. Wang et al. 2018a, b). The principle is based on the observation that molecules in bulk, non-viscous fluids exhibit fast, isotropic motion, resulting in a $T_1/T_2$ ratio close to unity. However, as molecular motion becomes slower or anisotropic, such as in highly viscous fluids or fluids strongly interacting with pore surfaces (i.e., wetting fluids), the $T_1/T_2$ ratio tends to be greater than one. This makes the $T_1/T_2$ ratio a better predictor of wettability, especially when diffusion relaxation is significant. A notable challenge with this technique arises when dealing with heavy oils, particularly those containing asphaltenes, as their inherent high viscosity can yield $T_1/T_2$ ratios greater than one, potentially complicating wettability interpretation (Valori et al. 2017; Valori and Nicot 2019). Nevertheless, NMR holds great potential for in-situ wettability evaluation (Valori et al. 2018), with ongoing research employing simulation tools to investigate NMR responses in multiphase rock core conditions (Al-Muthana et al. 2012; Looyestijn 2008; Mohnke et al. 2015; Wang et al. 2018a, b).

Furthermore, as discussed in the Pore Size Distribution subsection, the decomposition of the $T_2$ spectrum into distinct components provides a powerful method for identifying fluid properties (Zhong Jibin et al.), even in complex low-porosity and low-permeability reservoirs where fluid signatures often overlap. Each component, identified by fitting the $T_2$ spectrum with Gaussian functions, corresponds to different fluid environments such as clay-bound water, capillary-bound fluid, micropore fluid, and macropore fluid. Analysis of the free relaxation characteristics of crude oil and formation water, along with core-scale oil-water displacement experiments, allows for the precise identification of fluid types. For example, specific ranges of $T_2$ component spectral peaks can be defined for oil-bearing reservoirs, enabling accurate fluid identification. This advanced spectral decomposition significantly enhances the diagnostic capabilities of NMR in complex geological settings.

## 4. NMR in Enhanced Oil Recovery (EOR) Monitoring

Nuclear Magnetic Resonance (NMR) technology serves as a highly effective and versatile tool for characterizing and monitoring various Enhanced Oil Recovery (EOR) operations, both in laboratory settings and at the field scale [5, 109, 110, 7]. Its capability to provide detailed insights into fluid behavior within porous media allows for a granular understanding of displacement mechanisms, delineating oil and gas recovery from distinct pore systems within the rock matrix [5, 111, 12]. A primary objective of applying NMR in EOR is to screen various chemicals for their efficacy in applications such as carbon dioxide (CO₂), surfactant, and polymer flooding [5, 112, 113, 114]. Typically, the distribution of the $T_2$ relaxation time is employed to generate saturation profiles as a function of distance or along treated rock samples [109, 111, 115, 116].

Figure 9 illustrates the $T_2$ spectrum distributions of a core sample before and after various flooding scenarios, including water flood, CO₂-foam flood, and water-alternating-gas (WAG) injection. It clearly depicts how different EOR methods selectively impact oil saturation across various pore throat sizes. Initially, a significant portion of oil is present in moderate pore throats. After water flooding, the residual oil is primarily concentrated in smaller pore throats [117, 47, 16]. Subsequent CO₂-foam flooding effectively targets and recovers oil from these smaller pores [118, 119], while WAG is more efficient in larger pore throats [113]. The combined application of these methods demonstrates the highest overall recovery by addressing oil in a broader range of pore sizes. By acquiring $T_2$ relaxation time profiles before and after an EOR treatment, researchers can accurately quantify the remaining oil saturation and gain valuable information regarding its distribution within the reservoir [111, 109, 112, 116].

<center>Figure 9: Measured T2 spectrum distributions of core no. 1 before and after various floods. Adapted from Gao, Hui, et al., 2015 [16]. </center>

### 4.1 Chemical EOR Monitoring

In chemical EOR (CEOR) processes, NMR plays a crucial role in monitoring the progress of injected chemicals, particularly surfactants and polymers. The technique allows for the evaluation of changes in pore surface wettability, which are often induced by surfactant adsorption onto the rock matrix. This real-time monitoring capability is vital for optimizing CEOR performance. For instance, studies have utilized spatial $T_2$ profiles to track oil saturation during alkaline surfactant (AS) flooding in carbonate rocks. By co-injecting brine and AS under simulated reservoir conditions of high pressure and high temperature, NMR tools can effectively track the oil-AS interface, thereby enhancing the understanding of oil recovery mechanisms for different surfactant formulations. Furthermore, NMR analysis can help identify injectivity issues during AS flooding by pinpointing plugged pores where oil displacement by injected chemicals is hindered.

Low-field NMR tools have been instrumental in laboratory settings for monitoring oil saturation during water flooding and sulfonate-based nanosurfactant (NS) injection in high-permeability carbonate core samples [112, 111, 115, 16]. The primary goal of such NMR analyses is to elucidate the underlying oil recovery mechanisms [112, 111, 113, 120]. Experiments employing various injection schemes and soaking periods have demonstrated that sequential injection of nano-surfactant followed by water flooding can yield the highest oil recovery [112]. NMR analysis in these cases has revealed that NS injection significantly enhances oil production by mobilizing both trapped and adsorbed oil from the rock surface [112].

For polymer flooding, NMR offers a comprehensive approach for characterization at various stages [114, 121, 5]. This includes assessing changes in the polymer's chemical structure, tracking its movement within the porous media [116], and identifying instances of polymer plugging [122]. The ability to evaluate gelation parameters, such as gel strength and gelation time, in-situ using NMR measurements further underscores its utility [114, 116]. A significant advantage of NMR in polymer flooding operations is its non-destructive nature, which facilitates quick and reliable measurements, providing critical data for optimizing these complex processes [5, 67]. While extensive research has been conducted, the interaction between oil and injected fluids during polymer flooding remains an area requiring further in-depth NMR investigation to fully disclose the intricate rock and fluid behaviors.

### 4.2 Gas Injection EOR Monitoring

Gas injection EOR methods, particularly those involving CO₂ flooding, benefit significantly from NMR monitoring. The technique is employed to characterize hydrocarbon flow and pore structures in tight sands during EOR operations. By determining oil recovery and optimizing soaking periods during huff-n-puff (HnP) cycles, NMR provides valuable profiles of free fluid (FF), capillary bound fluid (CAF), and clay-bound fluid (CBF) based on $T_2$ relaxation times. This allows for detailed analysis of fluid displacements across different pore sizes and system pressures. For example, studies have shown that during the initial cycles of CO₂ injection, free fluid and a substantial portion of capillary bound fluid are recovered from small and medium pores, while clay-bound fluid remains largely unaffected. Such detailed saturation profiles are crucial for optimizing CO₂ injection strategies.

The interaction between injected CO₂ and reservoir minerals is another critical aspect that NMR helps to unravel [6, 68, 113, 85]. Different mineral compositions within tight cores can significantly influence tight oil recovery [123, 27, 97]. For instance, in illite-dominated cores, CO₂ may not efficiently extract oil from smaller pores when the injection pressure is below the minimum miscible pressure (MMP) [123, 124]. However, if the injection pressure surpasses the MMP, crude oil can be effectively recovered from both small and large pores [123, 124, 125]. Conversely, in montmorillonite-dominated cores, oil saturation in medium pores might increase upon CO₂ injection, while in quartz-dominated cores, oil residing in both small and large pores can be substantially recovered [123, 126]. These pore-scale insights into mineral-fluid interactions are fundamental to understanding the mechanisms of CO₂ flooding [113, 125, 123].

Beyond CO₂ injection, NMR is also applied to more complex gas injection schemes such as CO₂-foam flooding and water-alternating-gas (WAG) injection [125, 127, 5]. These advanced EOR methods aim to improve sweep efficiency and mobilize residual oil. NMR tests can quantitatively determine the initial and residual oil distributions, revealing how different flood schemes impact microscopic oil recovery [109, 111, 16, 117]. For example, after a water flood, residual oil is predominantly found in smaller pore throats [117, 113, 94, 92]. CO₂-foam flooding has demonstrated effectiveness in recovering residual oil from these smaller pore throats, whereas WAG is more efficient at recovering oil from larger pore throats. The synergistic combination of CO₂-foam flooding and WAG has been shown to provide the highest overall recovery by mobilizing oil from a wider range of pore sizes.

### 4.3 Thermal EOR Monitoring

Thermal EOR techniques, such as steam injection, are vital for heavy oil reservoirs where high viscosity hinders conventional production. While the provided references offer a general mention of NMR's application in thermal EOR, specific details on steam injection monitoring are limited. However, the broader utility of NMR in high-temperature environments is evident. Low-field NMR relaxometry, in conjunction with nonlinear least squares analysis, has been successfully employed for in-situ heavy oil viscosity prediction at elevated temperatures. This capability is paramount for understanding how thermal energy impacts oil properties and mobility within the reservoir, which is a core aspect of thermal EOR. NMR's sensitivity to fluid properties and phase behavior makes it an invaluable tool for assessing changes in oil viscosity and monitoring the overall effectiveness of thermal recovery processes. The ability to conduct these measurements under high-pressure and high-temperature conditions, representative of actual reservoir environments, is crucial for obtaining reliable and actionable data.

### 4.4 Techniques for Saturation and Fluid Distribution

NMR's adaptability stems from its diverse measurement techniques tailored for EOR applications. The $T_2$ distribution is a common approach for capturing changes in the rock porosity system and monitoring fluid saturations throughout EOR experiments.

Figure 10 illustrates a comparison between pore size distributions (PSD) obtained from NMR and those derived from Mercury Injection Capillary Pressure (MICP). This comparison is crucial in EOR monitoring as it provides a comprehensive understanding of the reservoir's pore architecture. While NMR offers insights into the distribution of fluids within the pore network, MICP provides details on the pore throat sizes. Correlating these two methods helps to accurately characterize the porous medium, which is vital for predicting how injected EOR fluids will flow, distribute, and displace oil. Understanding the pore structure aids in optimizing EOR strategies by matching fluid properties to the dominant pore sizes, thereby enhancing recovery efficiency.

<center>Figure 10: Comparison of NMR and MICP pore size distributions. Adapted from Elsayed, Mahmoud, et al., 2022 [5]. </center>

Beyond $T_2$, various other techniques are employed to determine oil saturation [67, 111, 109, 115], including chemical selective imaging [119, 116], complete signal suppression [112, 35], and paramagnetic doping [35, 50]. The choice of the most appropriate technique is contingent upon factors such as rock type, composition, and the specific chemical agents involved in the EOR process.

The Pulsed Field Gradient (PFG) NMR technique is particularly valuable for determining the diffusion coefficient of various fluids [128, 129, 130, 50]. This information can be utilized at early stages of EOR treatment to select suitable methods based on the pore network characteristics [5, 111, 112]. Diffusion measurements also aid in assessing pore coupling, which is critical for screening chemicals before their application in EOR [88, 90, 89]. For instance, in reservoirs with poor connectivity, chemicals that enhance pore connectivity might be preferred, while highly viscous fluids that necessitate high injection pressures could be excluded [100, 131, 85]. Mapping oil and water distributions in porous media through apparent diffusion coefficients is highly beneficial for designing and screening fluid systems for EOR [50, 35, 12].

<center>Figure 11: NMR T1-T2 maps of shale samples at different stages (native, imbibed, dried) for oil and brine portions. Adapted from Liu, Bo, et al., 2022 [132]. </center>

Figure 11 showcases NMR $T_1 - T_2$ maps, which are powerful tools for characterizing fluid phases (oil and brine) and their intricate behavior within shale samples under various EOR-relevant conditions, such as native, imbibed, and dried states. These 2D correlation maps provide a distinct separation of different fluid components based on their unique relaxation properties. For instance, by comparing the maps of oil and brine portions under native conditions (a and d), imbibed conditions (b and e), and dried conditions (c and f), researchers can precisely track fluid saturation changes, identify trapped fluids, and assess the effectiveness of imbibition processes. This detailed fluid characterization is essential for understanding the complex fluid-rock interactions in unconventional reservoirs and for optimizing EOR strategies to mobilize otherwise inaccessible hydrocarbons.

Magnetic Resonance Imaging (MRI) further augments NMR's capabilities by spatially resolving the distribution of fluid phases (water, oil, and gas) within the core samples [133, 134, 135, 50]. Laboratory-scale MRI is routinely used to assess the performance of different oil recovery strategies, including those involving chemicals (acids, polymers), supercritical CO₂, or miscible gases, prior to pilot operations in a reservoir [5, 109, 111, 113]. The ability to probe nuclei other than $^1$H, such as $^{23}$Na, allows for studies of miscible fluid injection, like analyzing oil recovery with varying salinity brine [68, 112]. Current research focuses on spatially resolved relaxation and diffusion measurements to generate wettability maps, an essential tool for core analysis, leading to a deeper understanding of oil recovery processes [77, 35, 76, 66]. Recent advancements in MRI techniques also enable the simultaneous measurement of rock structure (pore and grain size distributions) and fluid flow (flow propagators) on the same core plug [68, 12, 133, 136]. These laboratory-scale data are indispensable for refining models that predict the reservoir-scale effectiveness of EOR technologies, promising future insights into the intricate structure-transport interactions governing fluid flow in reservoirs [5, 109, 111].

<center>Figure 12: Schematic representation of a typical Nuclear Magnetic Resonance (NMR) integrated core flooding setup for Enhanced Oil Recovery (EOR) studies. </center>

Figure 12 illustrates a typical integrated core flooding setup with an NMR system, commonly used in laboratory-scale EOR studies. This schematic highlights the key components: a fluid injection system (comprising pumps and reservoirs) that supplies EOR fluids to a core holder containing the rock sample. The core holder is positioned within an NMR magnet, which is connected to an NMR spectrometer for data acquisition and analysis. An effluent collection system gathers fluids after they pass through the core. This integrated setup is critical for EOR research because it allows for real-time, non-invasive monitoring of fluid displacement, saturation changes, and fluid-rock interactions within the core sample under controlled conditions, directly correlating injection strategies with their microscopic effects on oil recovery.

### 4.5 Advantages and Limitations of NMR in EOR

A significant advantage of employing NMR for evaluating EOR operations is its capacity for detailed and continuous monitoring of remaining oil saturation, essential for optimizing performance. This capability extends to field applications, where NMR logging tools provide continuous in-situ oil saturation profiles due to their high acquisition speed. The integration of laboratory NMR coreflooding experiments with field logging data offers a comprehensive assessment of remaining oil saturation, thereby enhancing EOR strategies.

Despite these benefits, benchtop NMR instruments commonly used in laboratories present a key limitation: their typical design accommodates small core samples (2-4 inches). This restricts the analysis of longer core samples (6-20 inches), which are often preferred in coreflooding experiments to minimize capillary end effects and ensure more reliable results. A practical approach to overcome this involves cutting longer cores into smaller pieces and assembling composite cores for chemical flooding studies. Furthermore, while NMR has demonstrated broad utility in the energy sector, its application in EOR, particularly for a detailed understanding of rock-fluid interactions during polymer flooding, is still an evolving area with ongoing research needs. Future advancements are anticipated to bridge the gap between laboratory findings and field conditions, including the development of more efficient logging-while-drilling (LWD) hardware capable of operating effectively at high-pressure and high-temperature reservoir environments.

## 5. NMR in Unconventional Reservoirs

Unconventional reservoirs, encompassing shales, tight gas sands, and heavy oil/oil sands, present unique challenges for characterization due to their complex pore structures, varied fluid compositions, and low permeability [5, 69, 6, 34]. Nuclear Magnetic Resonance (NMR) has emerged as an indispensable tool for understanding these intricate systems, offering non-invasive and non-destructive analysis of critical petrophysical properties and fluid behavior [6, 34, 5]. Interpreting NMR signals in these formations is inherently more complex than in conventional reservoirs, primarily because the solid matrix and fluids exhibit distinct characteristics [34, 80, 137]. For instance, the hydrogen content within organic matter contributes significantly to the NMR signal in shales, making the resulting signal less dependent on lithology alone [34, 80, 138, 85]. This section delves into the specific applications of NMR across these diverse unconventional reservoir types, highlighting its utility in characterizing their unique properties.

### 5.1 Shale Characterization

Shale rocks, known for their fine-grained nature, high total organic content (typically exceeding $2\%$), and inherently low porosity and permeability [139, 140, 141, 34], present significant challenges for characterization. Their complex pore networks, comprising both organic and inorganic structures [142, 143, 94, 144], necessitate advanced analytical techniques like NMR [6, 34]. The inorganic component primarily consists of minerals such as silica, various clays, carbonates, and pyrite [140, 145]. These diverse pore types are visually exemplified in Figure 13, which illustrates the presence of organic pores within organic matter, intergranular and intragranular dissolution pores, intercrystalline pores in pyrite, and various micro-fractures, all contributing to the complex pore architecture of shales [142, 143, 94, 144]. Conversely, the organic fraction encompasses a spectrum of materials, including kerogen (porous, insoluble organics), bitumen (soluble, highly viscous organics), low viscosity oils, and natural gas [34, 85, 146, 147]. The precise quantification of these organic components is paramount for accurately assessing shale reservoir quality and maturity, playing a vital role in establishing the economic viability of these unconventional resources [79, 148, 149, 94].

<center>Figure 13: Organic pores, inorganic pores, and micro-fractures in shale reservoirs of the Wufeng Formation—Long1; sub-member in western Chongqing area. Adapted from Fu, Yonghong, et al., 2021 [155]. </center>

The relaxation dynamics observed in NMR experiments within shale rocks are notably intricate [6, 34]. In the inorganic pore structures, traditional relaxation models often apply, with surface relaxation being the dominant mechanism, analogous to conventional sandstones and carbonates. This phenomenon arises from the interaction of fluid molecules adsorbed onto pore surfaces with small quantities of paramagnetic metal species present at these surfaces [39, 27, 44, 26]. However, the relaxation mechanisms within the organic pore structures of shales are considerably more complex and remain a subject of ongoing discussion in the literature [34]. Several factors are understood to influence relaxation in these organic pores, including the thermal maturity of kerogen, where the solid proton density can significantly affect adsorbed surface relaxation [150, 36]. Furthermore, the nano-sized nature of kerogen pores introduces additional relaxation contributions due to molecular confinement effects [30, 74]. Diffusive coupling between organic and inorganic pore structures can also lead to an averaging of relaxation signals, potentially masking the presence of two distinct pore sizes, particularly when gas diffuses between these compartments [89, 88, 42, 34]. The origin of the organic matter itself also plays a crucial role, as it often contains paramagnetic metalloporphyrins such as Fe³⁺ and Mn²⁺ [151]. The presence and distribution of these paramagnetic minerals can substantially alter both longitudinal ($T_1$) and transverse ($T_2$) relaxation times, further complicating interpretation [44, 151, 26]. Lastly, the wetting characteristics of the pore structures are significant; inorganic pore structures are typically hydrophilic (water-wet), while organic pore structures are generally hydrophobic (oil-wet) [76, 152, 94].

For reliable characterization of shale rocks using NMR, understanding these relaxation considerations is vital. Two-dimensional (2D) NMR relaxometry, particularly $T_1 - T_2$ maps, offers a powerful approach for identifying and quantifying different fluid phases, including water, oil, and solid organics, within shale samples [65, 153, 154, 132]. This technique is superior to one-dimensional $T_2$ distributions alone, as the latter can suffer from overlapping signals from oil, water, organic matter, and bound water, making differentiation challenging in unconventional cores. Figure 11 illustrates a global $T_1/T_2$ map for fluid typing in shale rocks, which guides the interpretation of fluid behavior. Free bulk fluids typically exhibit long relaxation times, with $T_1$ and $T_2$ values being nearly identical, thus plotting close to the $T_1/T_2 = 1$ diagonal line [51, 18, 35]. Any signal detected in the region where $T_1$ is less than $T_2$ is considered physically unreasonable and often attributed to experimental artifacts. In $T_1 - T_2$ maps, the water region tends to be concentrated along the diagonal line, indicating a $T_1/T_2$ ratio close to unity. In contrast, the oil region often displays a more complex and broad distribution of $T_1/T_2$ ratios, typically ranging from 5 to 100, influenced by factors such as pore-size distribution, connectivity, organic matter content, and wettability of the shales [85, 35, 76, 94]. Molecules of heavy or solid organics, being less mobile, are characterized by very short $T_2$ values and relatively long $T_1$ values, leading to high $T_1/T_2$ ratios, which occupy distinct regions on the map [147, 154, 150, 85].

The application of higher-field NMR measurements, such as those at 19 MHz or 22 MHz, has significantly advanced shale characterization compared to traditional 2 MHz low-field NMR [68, 156, 27]. These higher frequencies offer improved signal-to-noise ratios (SNR), reduce detection times, and crucially, enhance the capability to measure bound water and solid organics, which typically exhibit very short $T_2$ relaxation times [68, 156, 27]. Higher-field NMR boasts a sensitivity 30 to 50 times greater than low-field NMR in distinguishing hydrogen signals from various proton populations, including hydroxyl groups, solid organic matter, oil/bitumen, and water [138, 137, 154, 65]. This enhanced resolution is critical for detailed analysis of shale oil occurrence characteristics and its mobility, which are often poorly understood despite their importance for resource evaluation and optimization of exploration targets [85, 80, 94, 79].

Beyond fluid typing, NMR is instrumental in characterizing the complex pore structure of shales [67, 69, 149]. Shales possess a significant proportion of pores in the nanometer size range, which are beyond the resolution limits of conventional microscopy techniques [74, 157, 158]. NMR, combined with methods like Mercury Intrusion Porosimetry (MIP) or nitrogen adsorption, provides a comprehensive understanding of pore size distribution [93, 74, 159, 92]. Fractal analysis, which accounts for the self-similar and irregular nature of pore structures, has also been successfully applied using NMR $T_2$ spectra to quantify micropores, mesopores, and macropores in tight formations [93, 159, 29]. Furthermore, NMR techniques can assess the wettability of shales [76, 152, 94], revealing that illitic shales tend to be strongly water-wet, while the presence of kaolinitic clays can impart an oil-wet tendency, with significant implications for oil and gas recovery strategies [160].

### 5.2 Tight Gas Sands

Tight gas sandstones are characterized by exceptionally narrow pore throats, intricate pore structures, and pronounced heterogeneity at both microscopic and macroscopic scales. The ultra-low permeability inherent to these formations poses significant challenges for effective exploitation. In such reservoirs, the pore structure exerts a dominant influence on reservoir properties and fluid seepage characteristics, directly impacting the distribution, storage capacity, and ultimate recovery of natural gas. Consequently, quantitative studies of microscopic pore structures in tight sandstones have garnered substantial attention.

NMR stands out as a rapid, non-destructive, and repeatable method for characterizing the pore structures and fluid states within these challenging porous media. It facilitates high-precision measurement of micro-nano scale porosity in tight rocks and supports the high-throughput continuous testing required for efficient reservoir evaluation. The NMR transverse relaxation time ($T_2$) spectrum is commonly employed to infer the pore size distribution. For tight oil sandstones, the relationship between NMR $T_2$ and pore size is often found to be more accurately described by power function relations rather than linear ones, which is crucial for deriving precise pore size distributions from NMR data. Advances in methodology include the development of new approaches to determine the conversion factor from NMR $T_2$ distribution to pore size distribution, yielding results that are more suitable for tight sandstones when compared with traditional methods like MIP. The integration of NMR with MIP, which provides capillary pressure curves and a wide range of pore sizes, allows for an advanced fractal analysis of the complex pore networks. The fractal dimensions calculated from the NMR $T_2$ spectrum have proven effective for petrophysical properties analysis, providing a quantitative measure of the complexity and irregularity of the pore space.

Furthermore, NMR measurements are instrumental in estimating permeability in tight gas sands, with models often integrating grain size information to improve prediction accuracy, as grain size is a critical factor influencing NMR response and permeability. Beyond static properties, NMR is also applied to evaluate movable fluid distribution (MFD) in tight reservoirs. This involves combining NMR tests with centrifugation experiments to analyze fluid changes within the pores. Through such studies, an optimum centrifugal force can be determined to establish the threshold radius for fluid flow. Research indicates that movable fluids in tight sandstones are predominantly controlled by pore throats with radii smaller than $1 \mu$m, particularly those ranging from $0.3$ to $1 \mu$m. These movable fluids are primarily stored in pores corresponding to the movable peak of a bimodal NMR $T_2$ distribution, typically with radii between $10$ and $100 \mu$m, which represent residual interparticle pores and dissolution pores. While direct studies on hydraulic fracturing impact on tight gas sands specifically using NMR are less detailed in the provided context, the general application of NMR to monitor changes in shale microstructure during EOR operations, such as Huff-n-Puff, suggests its potential for assessing alterations in pore throat size and surface area induced by fracturing, which is a key process in unconventional gas recovery. This ability to detect subtle changes in pore geometry underscores NMR's value in understanding the effects of reservoir stimulation.

### 5.3 Heavy Oil and Oil Sands

Nuclear Magnetic Resonance also plays a critical role in the characterization and monitoring of heavy oil and oil sands reservoirs, which present distinct challenges due to the high viscosity and often low mobility of their hydrocarbon components. NMR $T_1 - T_2$ spectra have been successfully applied in heavy oil reservoirs to gain insights into their complex fluid systems. A significant application of low-field NMR relaxometry in this domain is the in-situ prediction of heavy oil viscosity, even at elevated temperatures, which is crucial for optimizing production strategies given the strong temperature dependence of heavy oil viscosity. Moreover, NMR tools have been utilized for detailed heavy oil characterization during the precise placement of horizontal injectors at tar/oil interfaces, providing valuable information for field development and preventing issues like tar mat detection.

The ability of NMR to distinguish different fluid types, such as free water, capillary-bound water, and various oil fractions, is particularly advantageous in heavy oil systems. This fluid typing capability is fundamental for understanding the distribution and mobility of hydrocarbons within the reservoir pore space. Beyond static characterization, NMR is increasingly employed for monitoring Enhanced Oil Recovery (EOR) techniques in heavy oil and oil sands, including thermal-EOR processes such as Steam-Assisted Gravity Drainage (SAGD). NMR provides a unique capability to define the recovery of oil and gas from different pore systems within the rocks, offering a more granular understanding compared to macroscopic techniques that only assess bulk recovery. For instance, NMR allows for the monitoring of oil saturation using spatial $T_1$ profiles during chemical flooding operations, helping to track the oil-chemical interface and understand recovery mechanisms. This includes assessing injectivity problems by locating plugged pores where oil displacement is hindered. The technique has also been used to monitor oil saturation during water flooding and nano-surfactant injection, revealing how such treatments can mobilize trapped and adsorbed oil from rock surfaces. Furthermore, NMR facilitates the characterization of hydrocarbon flow and pore structures during EOR operations, such as CO₂ injection, by providing profiles of free fluid, capillary bound fluid, and clay-bound fluid based on $T_2$ relaxation times. This allows for detailed analysis of fluid displacements across different pore sizes and pressures, aiding in the optimization of injection schemes. The utility of NMR extends to observing microstructural changes in the reservoir rock induced by EOR treatments, such as an increase in pore throat size and pore surface area following gas injection into shales, which provides critical insights for the design and application of EOR operations in tight and unconventional reservoirs. The characterization of oilfield emulsions, frequently associated with heavy oil production, through techniques like Pulsed Field Gradient (PFG) NMR, which measures restricted diffusion to determine emulsion droplet size, further demonstrates NMR's versatility in addressing production challenges in these complex systems.

## 6. Advanced NMR Techniques and Future Trends

Nuclear Magnetic Resonance (NMR) technology continues to evolve, with significant advancements in multi-dimensional techniques [62, 50, 67], high-field instrumentation [68], and sophisticated imaging capabilities [4, 133]. These developments aim to overcome limitations of conventional NMR methods, offering enhanced resolution, sensitivity, and comprehensive insights into complex porous media and fluid systems. The ongoing research and development in these areas are crucial for pushing the boundaries of NMR applications, particularly in the oil and gas industry [67, 8, 7, 82], where detailed understanding of reservoir properties and fluid behavior is paramount [161, 131, 79, 75]. Furthermore, the integration of cutting-edge computational methods [50, 162] and hardware innovations [7, 163, 164, 165] is shaping the future landscape of NMR, promising more efficient and accurate characterization tools.

### 6.1 Multi-Dimensional NMR

Multi-dimensional NMR techniques represent a significant leap forward from traditional one-dimensional measurements, providing richer and more detailed information about fluid properties and pore structures within complex systems. Among these, $T_1 - T_2$ correlation spectroscopy and diffusion-weighted NMR are particularly prominent for their ability to resolve overlapping signals and characterize intricate transport phenomena.

$T_1 - T_2$ correlation spectroscopy, often referred to as 2D NMR, leverages the distinct relaxation behaviors of different fluids and their interactions with pore surfaces to provide a more definitive fluid typing and a more accurate characterization of pore size distribution [62, 57, 11, 85]. Unlike 1D $T_2$ spectra, which can suffer from overlapping signals from various fluid phases (e.g., oil and water in low-porosity reservoirs), 2D NMR maps allow for improved separation of these components [62, 57, 11, 85]. For instance, studies have demonstrated the effectiveness of $T_1 - T_2$ maps in identifying and quantifying water, oil, and solid organics in shale samples, offering both qualitative and quantitative insights (Fleury and Romero-Sarmiento, 2016; Khatibi et al., 2019)[85]. The $T_1/T_2$ ratio, a key parameter derived from these measurements, has also proven to be a more robust indicator for wettability prediction, especially when diffusion relaxation effects are significant (Katika et al., 2017; Valori et al., 2017)[78, 51, 18, 166]. This is because 2D NMR provides a clearer separation of fluid populations, allowing for a more accurate assessment of how molecular motion and surface interactions influence the $T_1/T_2$ ratio, which deviates from unity for viscous or wetting fluids.

Figure 14 exemplifies the utility of $T_1 - T_2$ maps in monitoring dynamic processes within porous media, showing the changes in fluid distribution within a shale sample during imbibition over several days. Such maps visually represent the evolution of water and oil signals, providing direct insights into fluid movement and saturation changes within different pore environments over time.

<center>Figure 14: NMR T1-T2 maps of the shale sample from 2107.59 m depth at different imbibed stages of 0 days, 5 days, 14 days and 21 days. Adapted from Liu, Bo, et al., 2022 [132]. </center>

Beyond $T_1 - T_2$ correlation, $T_2 - D$ NMR techniques, which map transverse relaxation time against the diffusion coefficient, offer another powerful approach for fluid characterization and separation [62, 35, 57, 66]. This method is particularly effective for distinguishing fluids based on their molecular mobility, which is influenced by both fluid viscosity and pore confinement [35, 136, 131, 130]. The improved fluid separation achieved with $T_2 - D$ NMR also contributes to more accurate calculations of effective surface relaxivities of fluids (Minh et al., 2015). The ability of 2D NMR techniques to better distinguish between different fluid phases and their confinement states is invaluable for understanding fluid distribution in heterogeneous porous media and for evaluating enhanced oil recovery (EOR) processes [35, 113, 85, 50].

Diffusion-weighted NMR, particularly techniques involving Pulsed Field Gradients (PFG), provides critical information about fluid mobility and pore connectivity [167, 128, 130, 67]. The PFG NMR method is non-invasive and allows for the determination of diffusion coefficients for various fluids within porous media (Elsayed et al., 2021b; Johnson, 1999; Willis et al., 2016) [136, 168, 129, 81]. This is particularly useful in the early stages of EOR treatments, as it can help in selecting suitable methods based on the pore network characteristics. For example, in reservoirs with poor connectivity, chemicals designed to improve pore connectivity might be favored, while highly viscous fluids requiring high injection pressures could be excluded (Elsayed et al., 2021b) [112, 67, 100, 169]. Diffusion measurements can also assess pore coupling phenomena, where fluid molecules can exchange between pores of different sizes, especially in multi-modal pore systems like those found in carbonates [88, 89, 90, 42]. Understanding these diffusion coupling effects is essential for accurately estimating pore sizes from NMR relaxation times (Johnson and Schwartz, 2014; Yu et al., 2019) [90, 40, 29, 26]. The presence of internal magnetic field gradients, arising from susceptibility differences at fluid-solid interfaces, can significantly influence diffusion measurements and relaxation times (Hürlimann, 1998; Mitchell et al., 2010; Tandon and Heidari, 2018) [29, 26, 27, 39]. Advanced PFG techniques and numerical simulations are being developed to account for and utilize these internal gradients for more precise characterization of porous media (Connolly et al., 2019; Zhang et al., 1998) [50, 90, 26, 129]. The mapping of oil and water distributions in porous media using apparent diffusion coefficients is another powerful application, aiding in the design and screening of various fluid systems for EOR applications (Elsayed et al., 2021b) [50, 81, 82, 170].

### 6.2 High-Field NMR

The choice of magnetic field strength in NMR experiments significantly impacts the signal-to-noise ratio (SNR), resolution, and the types of phenomena that can be observed. High-field NMR spectrometers, characterized by very strong magnetic fields (typically exceeding 3 Tesla), offer distinct advantages for certain applications, particularly in the detailed characterization of complex fluid systems and heterogeneous rock structures.

The primary advantage of high-field NMR lies in its enhanced sensitivity and resolution. A stronger magnetic field generally leads to a higher polarization of nuclear spins, resulting in a significantly stronger NMR signal and thus a higher SNR (Hoult and Richards, 1976; Mitchell et al., 2013). This improved SNR allows for shorter acquisition times, reducing the overall experimental duration, and enables the detection of weaker signals from less abundant nuclei or smaller sample volumes. For instance, in the context of unconventional reservoirs, higher field NMR systems (e.g., 22 MHz compared to the more common 2 MHz low-field systems) can bring substantial benefits in terms of SNR and reduced detection time, while also providing the capability to measure bound water and solid organics that have very short $T_2$ relaxation times [156, 69, 80, 76]. This is crucial for characterizing the extremely small pores and complex organic matter in tight shales [74, 150, 79, 26].

<center>Figure 15: Field dependence of 50% suspension of yeast cells in their stationary phase of growth. (A) Spectrum taken at 40.5 MHz by pulsed FT NMR; (B) Spectrum taken at 145.7 MHz by pulsed FT NMR; (C) Same spectrum as B, after turning up the gain a factor of 8. Adapted from Salhany, J M, et al., 1975 [171]. </center>

The benefits of high-field systems are visually demonstrated in Figure 15, which illustrates how increasing the magnetic field strength from 40.5 MHz to 145.7 MHz dramatically improves the spectral resolution and signal definition, allowing for clearer distinction of molecular components. Although this specific example is from a biological system, the principle of enhanced resolution and sensitivity directly translates to the characterization of complex fluid mixtures and pore environments in petroleum applications.

The enhanced resolution of high-field NMR allows for finer discrimination of chemical shifts and relaxation times, which is essential for studying complex fluid mixtures, such as crude oils with varying compositions, or fluids within highly heterogeneous pore networks [27, 68, 100]. This capability can provide detailed insights into the molecular dynamics of fluids confined within pores, as well as their interactions with different mineral surfaces [88, 37, 18]. For example, high-field NMR can be used to probe different nuclei beyond $^1$H, such as $^{23}$Na, to monitor miscible fluid injection processes and analyze oil recovery with varying salinity brine (Yang and Kausik, 2016). The ability to acquire high-resolution spectra and relaxation data at high fields opens avenues for more sophisticated characterization of fluid-rock interactions, which is critical for optimizing various petroleum engineering applications, including EOR and formation damage assessment [67, 111, 112, 172].

### 6.3 NMR Imaging (MRI)

Nuclear Magnetic Resonance Imaging (MRI) extends the capabilities of traditional NMR by providing spatially resolved information, allowing for the visualization of fluid distribution and flow pathways within porous media. This non-invasive technique is invaluable for understanding complex phenomena that occur at the pore scale and for assessing macroscopic fluid transport in rock samples.

At the laboratory scale, MRI is extensively utilized to provide spatially resolved insights into the performance of various oil recovery strategies before their field implementation [67, 109, 110, 111]. Unlike bulk measurements, MRI directly visualizes the dynamic distribution of fluid phases—water, oil, and gas—within the rock matrix during processes such as chemical injections (e.g., acids, polymers) [112, 121], supercritical CO₂ flooding, or miscible gas injection [119, 173, 174, 113] (Lai et al., 2020; Li et al., 2017; Zhao et al., 2020a, b). This capability is particularly important for understanding the mechanisms of improved oil recovery (IOR) and EOR techniques, as it allows for direct observation of displacement fronts, identification of bypassed oil, and visualization of the impact of treatments on the pore structure at a macroscopic level [67, 109, 111, 135].

Beyond fluid distribution, recent advancements in MRI techniques enable the measurement of rock structure, including pore and grain size distributions, and fluid flow properties, such as flow propagators, on the same core-plug (Karlsons et al., 2021). This integrated approach provides a more comprehensive understanding of the structure-transport interactions that govern fluid flow in reservoirs. Spatially resolved relaxation and diffusion measurements, facilitated by MRI, are also crucial for generating wettability maps, which are key tools for core analysis and lead to a better understanding of oil recovery processes (Karlsons et al., 2021). These laboratory-scale data are vital for enhancing predictive models that forecast the reservoir-scale effectiveness of oil recovery technologies.

<center>Figure 16: Conceptual representation of a T1–T2 correlation map, illustrating how different fluid types (bound water, free water, light oil, heavy oil) can be distinguished based on their distinct relaxation characteristics. The diagonal line represents the ideal T1 = T2 condition, with deviations indicating varying degrees of molecular restriction and surface interaction. This visualization greatly aids in fluid typing and understanding pore environments. </center>

The application of MRI is not limited to laboratory core analysis. Surface Nuclear Magnetic Resonance (SNMR), also known as Magnetic Resonance Sounding (MRS), is a surface-based geophysical method that utilizes NMR principles to directly map groundwater aquifers without the need for boreholes. While earlier SNMR practices primarily relied on 1D inversion strategies, recent developments have introduced very fast 2D SNMR tomographic inversion schemes, which can more accurately reconstruct complex subsurface water distributions (Hertrich et al., 2007; Legchenko et al., 2011). The capabilities of SNMR are further exemplified by Figure 17, which presents a 3D tomographic reconstruction of water distribution in a glacier. This figure demonstrates SNMR's power to map subsurface water bodies in a non-invasive manner, providing crucial spatial information about aquifer geometry and water content, which is directly translatable to groundwater exploration and management in various geological settings. This capability to obtain NMR measurements from the Earth's surface revolutionizes noninvasive evaluation of groundwater resources, providing a direct link to the presence and amount of water in the pore space through the detection of hydrogen nuclei magnetization (Knight et al., 2016).

<center>Figure 17: West-East cross-section of the water distribution derived from 3D-SNMR measurements. Black columns show boreholes that did not detect water; pink columns show boreholes that intersected water-filled caverns. Adapted from Legchenko, A, et al., 2011 [175]. </center>

These advancements underscore the potential of MRI to provide crucial, spatially resolved data for both reservoir characterization and environmental applications.

### 6.4 Future Directions in NMR Research

The field of Nuclear Magnetic Resonance is dynamic, with ongoing research focused on enhancing its capabilities and expanding its applications across various scientific and industrial domains. Several key areas are poised for significant advancements, promising to revolutionize how we understand and characterize porous media and fluid systems.

One critical area of future development is the optimization and modification of NMR pulse sequences and data processing techniques. The current standard experiments, such as 2D NMR diffusion measurements, can be lengthy, which poses challenges for operational efficiency, especially in field applications. Developing robust numerical simulation models that account for all relaxation phenomena, including the effects of bulk and internal gradients, is a promising avenue to examine various scenarios and reduce the need for time-consuming physical experiments. Furthermore, enhancing the signal-to-noise ratio (SNR) is a key parameter for obtaining accurate measurements, and continuous efforts are directed towards designing pulse sequences that maximize signal acquisition while minimizing noise (Chen et al., 2018b; Sun et al., 2020). This involves sophisticated inversion algorithms and data reconstruction methods that can extract maximum information from acquired signals, even under challenging conditions.

Another vital direction involves the integration of NMR data with other geophysical and petrophysical measurements. While NMR provides unique insights into rock-fluid interactions, its full potential is realized when combined with complementary data sources. For instance, cross-validation between laboratory rock core measurements and field logging data is essential for building reliable predictive models for reservoir properties. Efforts to accurately overlap Mercury Intrusion Capillary Pressure (MICP) and NMR relaxation distributions, often by utilizing advanced imaging techniques like X-ray micro-computed tomography (μCT), aim to better evaluate surface relaxivity and pore size distributions (Connolly et al., 2019; Lyu et al., 2020). This multi-modal data integration approach helps to overcome the inherent limitations of individual techniques and provides a more comprehensive understanding of complex reservoir characteristics.

The application of machine learning (ML) and artificial intelligence (AI) is rapidly emerging as a transformative force in NMR data interpretation. Given the large volumes of complex, multi-dimensional data generated by advanced NMR experiments, conventional analysis methods can be time-consuming and may not fully capture subtle correlations. While specific applications of ML/AI to NMR in the context of this review's provided literature are still emerging, the overarching need for "accurate predictions" and "robust numerical simulation models" in reservoir characterization strongly points towards the increasing adoption of these advanced computational methods. ML and AI algorithms can be trained to recognize patterns, classify fluid types, predict petrophysical properties, and even automate the inversion of NMR relaxation data, leading to faster and more accurate interpretations. These technologies hold the potential to significantly streamline the NMR workflow, from data acquisition to final interpretation, making NMR a more accessible and powerful tool for reservoir characterization and monitoring.

Finally, the development of more efficient Logging While Drilling (LWD) hardware is crucial for bringing laboratory findings to field-scale applications. LWD-NMR tools have shown great potential in providing real-time information about formation lithology, fluid viscosity, and pore systems, aiding in geosteering and optimizing well placement (Jachmann et al., 2020; Prammer et al., 2000a; Sun et al., 2020). Future research will focus on improving the robustness, resolution, and data acquisition speed of these downhole tools to mimic the accuracy and detail achievable in laboratory settings, thereby providing more reliable real-time insights during drilling operations. These combined efforts in technique refinement, data integration, computational intelligence, and hardware innovation are paving the way for NMR to become an even more indispensable tool in the energy industry and beyond.

## 7. Conclusion

Nuclear Magnetic Resonance (NMR) technology has emerged as an indispensable tool across various facets of the oil and gas industry, revolutionizing the understanding and characterization of porous media and fluid behavior from laboratory to field scales [109, 176, 68]. This comprehensive survey has elucidated the fundamental principles governing NMR measurements, its diverse applications in petrophysics [82, 81, 70, 71], enhanced oil recovery (EOR) monitoring [109, 114], and unconventional reservoir characterization [69, 177, 80, 94], alongside its significant role in logging while drilling (LWD) operations and geosteering [7, 165, 161, 178]. The unique capabilities of NMR provide insights that are often unattainable or less precise with conventional methods, thereby enhancing decision-making in exploration, development, and production phases.

The profound impact of NMR stems from its ability to collectively provide a comprehensive and non-invasive characterization of reservoir rocks and their contained fluids [7, 8, 82, 68]. Unlike traditional techniques that may be destructive, lithology-dependent, or limited to bulk measurements, NMR offers a direct, lithology-independent probe of pore structure, fluid types, and their interactions. This directness is particularly valuable in complex geological settings, such as carbonates with intricate pore networks or shales with mixed wettability and nano-scale porosity, where conventional tools often fall short. In petrophysics, NMR's capacity to quantify porosity, derive pore size distributions, estimate permeability, and assess fluid saturation and wettability provides a foundational understanding of reservoir quality and producibility. Its application extends beyond static characterization to dynamic monitoring, proving essential for evaluating and optimizing EOR processes by revealing how injected fluids interact with the rock matrix and mobilize oil from different pore systems at the pore scale. In unconventional reservoirs, NMR's sensitivity to hydrogen in both fluids and organic matter, combined with advanced 2D techniques, allows for unparalleled insights into fluid storage, maturity, and producibility, which are critical for economic assessments.

However, the application of NMR is not without its challenges. Internal magnetic field gradients, arising from susceptibility contrasts between rock matrix and fluids, can complicate the interpretation of relaxation times and diffusion measurements, especially in rocks with paramagnetic minerals. Pore coupling effects, where diffusive exchange between pores averages relaxation times, can obscure true pore size distributions in rocks with multimodal pore systems. For low-porosity and low-permeability unconventional reservoirs, low SNR and overlapping fluid signals remain significant hurdles, demanding advanced acquisition and processing techniques. Another challenge pertains to the interpretation of wettability using $T_1/T_2$ ratios, especially in the presence of heavy oils or asphaltenes, where the ratio may deviate from expected values due to fluid properties rather than solely surface interactions. For low-porosity and low-permeability reservoirs, traditional NMR fluid identification methods can suffer from low accuracy due to significant overlap in the $T_2$ spectra of oil and water. This overlap makes it difficult to unambiguously differentiate between oil and water signals, especially in tight formations where fluid mobility and pore sizes are restricted. In field applications, LWD-NMR tools, while providing real-time data, can be influenced by drilling fluids in the borehole and radial vibrations during drilling, which may affect the reliability and accuracy of $T_2$ measurements. The development of robust numerical simulation models that account for all relaxation phenomena, including bulk and internal gradient effects, is crucial for saving time in lengthy experiments and improving predictive capabilities.

Looking ahead, the future of NMR technology in the oil and gas industry is poised for continuous innovation and broader application. A key direction involves the further development and widespread adoption of multi-dimensional NMR techniques, such as $T_1 - T_2$ and diffusion-$T_2$ maps, which offer superior fluid separation and provide more comprehensive information on pore connectivity and wettability. Enhancing the signal-to-noise ratio (SNR) of collected data, particularly during LWD operations, will be vital for improving measurement accuracy and reducing acquisition time, thereby lowering operational costs. Research efforts are also focused on developing more efficient LWD hardware that can mimic laboratory results, enabling more precise real-time formation evaluation downhole. To fully unlock the potential of these advanced NMR techniques and address complex reservoir problems comprehensively, fostering robust interdisciplinary collaboration among physicists, engineers, and geoscientists will be increasingly crucial.

Moreover, there is a growing need for NMR measurements to be conducted under reservoir-representative conditions, including high pressure and high temperature, using longer core samples to provide more reliable and scalable results for EOR evaluations. Investigations into the rock-fluid interactions during advanced EOR methods, such as polymer flooding, using NMR are also critical. The integration of NMR data with other advanced characterization techniques, such as X-ray computed tomography (CT) scanning, mercury intrusion porosimetry (MIP), and electrical resistivity measurements, will lead to a more holistic understanding of reservoir properties. The exploration of NMR applications involving nuclei other than hydrogen, such as carbon-13 and sodium-23, also holds promise for addressing specific questions related to fluid composition and rock-fluid interactions. In conclusion, NMR technology is not static but continuously evolving, with ongoing advancements poised to further solidify its role as an essential tool for unraveling subsurface complexity and optimizing hydrocarbon recovery in the years to come.

## 8. References

[1] Komal Zia et al. "Nuclear Magnetic Resonance Spectroscopy for Medical and Dental Applications: A Comprehensive Review". In: *European Journal of Dentistry* 13.01 (2019), 124-128. DOI: 10.1055/s-0039-1688654.

[2] Joseph P. Hornak. "Teaching NMR Using Online Textbooks". In: *Molecules* 4.12 (1999), 353-365. DOI: 10.3390/41200353.

[3] William S. Price. *NMR Studies of Translational Motion: Principles and Applications*. Cambridge University Press, 2009. DOI: 10.1017/cbo9780511770487.

[4] Paul T Callaghan. *Principles of Nuclear Magnetic Resonance Microscopy*. Oxford University Press, 1991. DOI: 10.1093/oso/9780198539445.001.0001.

[5] Mahmoud Elsayed et al. "A review on the applications of nuclear magnetic resonance (NMR) in the oil and gas industry: laboratory and field-scale measurements". In: *Journal of Petroleum Exploration and Production Technology* 12.10 (2022), 2747-2784. DOI: 10.1007/s13202-022-01476-3.

[6] Jian-Chun Guo et al. "Advances in low-field nuclear magnetic resonance (NMR) technologies applied for characterization of pore space inside rocks: a critical review". In: *Petroleum Science* 17.5 (2020), 1281-1297. DOI: 10.1007/s12182-020-00488-0.

[7] Kirill Kuptsov, Roger Griffiths, and David Maggs. "Technology Update: Magnetic Resonance-While-Drilling System Improves Understanding of Complex Reservoirs". In: *Journal of Petroleum Technology* 67.12 (2015), 26-29. DOI: 10.2118/1215-0026-jpt.

[8] Gabor Hursan, Andre Silva, and Mohamed L. Zeghlache. "Evaluation and Development of Complex Elastic Reservoirs using NMR". In: *SPE Annual Technical Conference and Exhibition*. SPE, 2016. DOI: 10.2118/181525-ms.

[9] Stefan Menger and Manfred Prammer. "A New Algorithm for Analysis of NMR Logging Data". In: *SPE Annual Technical Conference and Exhibition*. SPE, 1998. DOI: 10.2118/49013-ms.

[10] Wei Shao and Ron Balliet. "NMR Logging Data Processing". In: *Petrophysics* 63.3 (2022), 300-338. DOI: 10.30632/pjv63n3-2022a3.

[11] Jiangfeng Guo et al. "Nuclear Magnetic Resonance T1-T2 Spectra in Heavy Oil Reservoirs". In: *Energies* 12.12 (2019), 2415. DOI: 10.3390/en12122415.

[12] Hsie-Keng Liaw et al. "Characterization of fluid distributions in porous media by NMR techniques". In: *AIChE Journal* 42.2 (1996), 538-546. DOI: 10.1002/aic.690420223.

[13] Matej Hriberšek. "Predgovor". In: *Clotho* 6.1 (2024), 7-8. DOI: 10.4312/clotho.6.1.7-8.

[14] R.J.S. Brown and B.W. Gamson. "Nuclear Magnetism Logging". In: *Transactions of the AIME* 219.01 (1960), 201-209. DOI: 10.2118/1305-g.

[15] M. Josh et al. "Laboratory characterisation of shale properties". In: *Journal of Petroleum Science and Engineering* 88-89 (2012), 107-124. DOI: 10.1016/j.petrol.2012.01.023.

[16] Hui Gao et al. "Impact of Secondary and Tertiary Floods on Microscopic Residual Oil Distribution in Medium- to-High Permeability Cores with NMR Technique". In: *Energy & Fuels* 29.8 (2015), 4721-4729. DOI: 10.1021/acs.energyfuels.5b00394.

[17] Morrel H. Cohen and Kenneth S. Mendelson. "Nuclear magnetic relaxation and the internal geometry of sedimentary rocks". In: *Journal of Applied Physics* 53.2 (1982), 1127-1135. DOI: 10.1063/1.330526.

[18] Carmine D'Agostino et al. "Interpretation of NMR Relaxation as a Tool for Characterising the Adsorption Strength of Liquids inside Porous Materials". In: *Chemistry - A European Journal* 20.40 (2014), 13009-13015. DOI: 10.1002/chem.201403139.

[19] Kenneth S. Mendelson. "Nuclear Magnetic Relaxation in Porous Media". In: *Journal of The Electrochemical Society* 133.3 (1986), 631-633. DOI: 10.1149/1.2108633.

[20] A. Timur. "Pulsed Nuclear Magnetic Resonance Studies of Porosity, Movable Fluid, and Permeability of Sandstones". In: *Journal of Petroleum Technology* 21.06 (1969), 775-786. DOI: 10.2118/2045-pa.

[21] James J. Howard, William E. Kenyon, and Chris Straley. "Proton Magnetic Resonance and Pore Size Variations in Reservoir Sandstones". In: *SPE Formation Evaluation* 8.03 (1993), 194-200. DOI: 10.2118/20600-pa.

[22] S. Davies and K. J. Packer. "Pore-size distributions from nuclear magnetic resonance spin-lattice relaxation measurements of fluid-saturated porous solids. I. Theory and simulation". In: *Journal of Applied Physics* 67.6 (1990), 3163-3170. DOI: 10.1063/1.345395.

[23] N. Bloembergen, E. M. Purcell, and R. V. Pound. "Relaxation Effects in Nuclear Magnetic Resonance Absorption". In: *Physical Review* 73.7 (1948), 679-712. DOI: 10.1103/physrev.73.679.

[24] H. A. Resing and H. C. Torrey. "Nuclear Spin Relaxation by Translational Diffusion. III. Spin-Spin Relaxation". In: *Physical Review* 131.3 (1963), 1102-1104. DOI: 10.1103/physrev.131.1102.

[25] B. Kanchibotla et al. "Transverse spin relaxation time in organic molecules". In: *Physical Review B* 78.19 (2008). DOI: 10.1103/physrevb.78.193306.

[26] Saurabh Tandon and Zoya Heidari. "Effect of Internal Magnetic-Field Gradients on Nuclear-Magnetic-Resonance Measurements and Nuclear-Magnetic-Resonance-Based Pore-Network Characterization". In: *SPE Reservoir Evaluation & Engineering* 21.03 (2018), 609-625. DOI: 10.2118/181532-pa.

[27] Stian Almenningen et al. "Effect of Mineral Composition on Transverse Relaxation Time Distributions and MR Imaging of Tight Rocks from Offshore Ireland". In: *Minerals* 10.3 (2020), 232. DOI: 10.3390/min10030232.

[28] Paul R.J. Connolly et al. "Simulation and experimental measurements of internal magnetic field gradients and NMR transverse relaxation times (T2) in sandstone rocks". In: *Journal of Petroleum Science and Engineering* 175 (2019), 985-997. DOI: 10.1016/j.petrol.2019.01.036.

[29] Hugh Daigle, Andrew Johnson, and Brittney Thomas. "Determining fractal dimension from nuclear magnetic resonance data in rocks with internal magnetic field gradients". In: *GEOPHYSICS* 79.6 (2014), D425-D431. DOI: 10.1190/geo2014-0325.1.

[30] J.-P. Korb, Shu Xu, and J. Jonas. "Confinement effects on dipolar relaxation by translational dynamics of liquids in porous silica glasses". In: *The Journal of Chemical Physics* 98.3 (1993), 2411-2422. DOI: 10.1063/1.464169.

[31] J.D. Loren and J.D. Robinson. "Relations Between Pore Size Fluid and Matrix Properties, and NML Measurements". In: *Society of Petroleum Engineers Journal* 10.03 (1970), 268-278. DOI: 10.2118/2529-pa.

[32] Yang Chen et al. "Experimental study on the rheological characteristics and viscosity-enhanced factors of super-viscous heavy oil". In: *Liquid and Gaseous Energy Resources* 3.2 (2023), 67-75. DOI: 10.21595/lger.2023.23660.

[33] K. Allsopp et al. "Determination of Oil and Water Compositions of Oil/Water Emulsions Using Low Field NMR Relaxometry". In: *Journal of Canadian Petroleum Technology* 40.07 (2001). DOI: 10.2118/01-07-05.

[34] Kathryn E. Washburn. "Relaxation mechanisms and shales". In: *Concepts in Magnetic Resonance Part A* 43A.3 (2014), 57-78. DOI: 10.1002/cmr.a.21302.

[35] R. Freedman et al. "Wettability, Saturation, and Viscosity From NMR Measurements". In: *SPE Journal* 8.04 (2003), 317-327. DOI: 10.2118/87340-pa.

[36] Saurabh Tandon and Zoya Heidari. "Improved Analysis of Nuclear-Magnetic-Resonance Measurements in Organic-Rich Mudrocks Through Experimental Quantification of the Hydrocarbon/Kerogen Intermolecular-Interfacial-Relaxation Mechanism". In: *SPE Journal* 25.05 (2020), 2547-2563. DOI: 10.2118/202480-pa.

[37] Kristina Keating and Rosemary Knight. "The effect of spatial variation in surface relaxivity on nuclear magnetic resonance relaxation rates". In: *GEOPHYSICS* 77.5 (2012), E365-E377. DOI: 10.1190/geo2011-0462.1.

[38] Peiqiang Zhao et al. "Nuclear magnetic resonance surface relaxivity and its advanced application in calculating pore size distributions". In: *Marine and Petroleum Geology* 111 (2020), 66-74. DOI: 10.1016/j.marpetgeo.2019.08.002.

[39] Elliot Grunewald and Rosemary Knight. "The effect of pore size and magnetic susceptibility on the surface NMR relaxation parameter". In: *Near Surface Geophysics* 9.2 (2010), 169-178. DOI: 10.3997/1873-0604.2010062.

[40] Z. R. Hinedi et al. "Quantification of microporosity by nuclear magnetic resonance relaxation of water imbibed in porous media". In: *Water Resources Research* 33.12 (1997), 2697-2704. DOI: 10.1029/97wr02408.

[41] John Doveton and Lynn Watney. "Textural and pore size analysis of carbonates from integrated core and nuclear magnetic resonance logging: An Arbuckle study". In: *Interpretation* 3.1 (2015), SA77-SA89. DOI: 10.1190/int-2014-0050.1.

[42] Heng Wang et al. "Low-Field Nuclear Magnetic Resonance Characterization of Carbonate and Sandstone Reservoirs From Rock Spring Uplift of Wyoming". In: *Journal of Geophysical Research: Solid Earth* 123.9 (2018), 7444-7460. DOI: 10.1029/2018jb015779.

[43] Carmine D'Agostino et al. "Effect of paramagnetic species on T1, T2 and T1/T2 NMR relaxation times of liquids in porous CuSO4/Al2O3". In: *RSC Advances* 7.57 (2017), 36163-36167. DOI: 10.1039/c7ra07165e.

[44] Kristina Keating and Rosemary Knight. "A laboratory study to determine the effect of iron oxides on proton NMR measurements". In: *GEOPHYSICS* 72.1 (2007), E27-E32. DOI: 10.1190/1.2399445.

[45] I. Foley, S.A. Farooqui, and R.L. Kleinberg. "Effect of Paramagnetic Ions on NMR Relaxation of Fluids at Solid Surfaces". In: *Journal of Magnetic Resonance, Series A* 123.1 (1996), 95-104. DOI: 10.1006/jmra.1996.0218.

[46] Vivek Anand and George J. Hirasaki. "Paramagnetic relaxation in sandstones: Distinguishing T1 and T2 dependence on surface relaxation, internal gradients and dependence on echo spacing". In: *Journal of Magnetic Resonance* 190.1 (2008), 68-85. DOI: 10.1016/j.jmr.2007.09.019.

[47] Paul R. J. Connolly et al. "Capillary trapping quantification in sandstones using NMR relaxometry". In: *Water Resources Research* 53.9 (2017), 7917-7932. DOI: 10.1002/2017wr020829.

[48] E. O. Stejskal and J. E. Tanner. "Spin Diffusion Measurements: Spin Echoes in the Presence of a Time-Dependent Field Gradient". In: *The Journal of Chemical Physics* 42.1 (1965), 288-292. DOI: 10.1063/1.1695690.

[49] J. Mitchell et al. "Nuclear magnetic resonance relaxation and diffusion in the presence of internal gradients: The effect of magnetic field strength". In: *Physical Review E* 81.2 (2010). DOI: 10.1103/physreve.81.026101.

[50] E. Toumelin, C. Torres-Verdin, and S. Chen. "Quantification of Multi-Phase Fluid Saturations in Complex Pore Geometries from Simulations of Nuclear Magnetic Resonance Measurements". In: *SPE Annual Technical Conference and Exhibition*. SPE, 2002. DOI: 10.2118/77399-ms.

[51] R.L. Kleinberg, S.A. Farooqui, and M.A. Horsfield. "T1/T2 Ratio and Frequency Dependence of NMR Relaxation in Porous Sedimentary Rocks". In: *Journal of Colloid and Interface Science* 158.1 (1993), 195-198. DOI: 10.1006/jcis.1993.1247.

[52] R.J.S. Brown and B.W. Gamson. "Nuclear Magnetism Logging". In: *Transactions of the AIME* 219.01 (1960), 201-209. DOI: 10.2118/1305-g.

[53] Erwin L. Hahn. "Spin Echoes". In: *Physics Today* 3.12 (1950), 21. DOI: 10.1063/1.3066708.

[54] Arnold L. Bloom. "Nuclear Induction in Inhomogeneous Fields". In: *Physical Review* 98.4 (1955), 1105-1111. DOI: 10.1103/physrev.98.1105.

[55] Rosemary Knight et al. "Field experiment provides ground truth for surface nuclear magnetic resonance measurement". In: *Geophysical Research Letters* 39.3 (2012). DOI: 10.1029/2011gl050167.

[56] W. E. Kenyon et al. "A Three-Part Study of NMR Longitudinal Relaxation Properties of Water-Saturated Sandstones". In: *SPE Formation Evaluation* 3.03 (1988), 622-636. DOI: 10.2118/15643-pa.

[57] A. E. English et al. "Quantitative Two-Dimensional time Correlation Relaxometry". In: *Magnetic Resonance in Medicine* 22.2 (1991), 425-434. DOI: 10.1002/mrm.1910220250.

[58] S. Meiboom and D. Gill. "Modified Spin-Echo Method for Measuring Nuclear Relaxation Times". In: *Review of Scientific Instruments* 29.8 (1958), 688-691. DOI: 10.1063/1.1716296.

[59] Rieko Ishim. "CPMG Relaxation Dispersion". In: *Protein Dynamics*. Humana Press, 2013, 29-49. DOI: 10.1007/978-1-62703-658-0_2.

[60] Martin Blanz et al. "Nuclear Magnetic Resonance Logging While Drilling (NMR-LWD): from an experiment to a day-to-day Service for the oil industry". In: *Diffusion Fundamentals* 14 (2010). DOI: 10.62721/diffusion-fundamentals.14.412.

[61] S. O. Chan. "Detailed echo-shape fitting in NMR spin-echo experiments". In: *The Journal of Chemical Physics* 62.6 (1975), 2031-2038. DOI: 10.1063/1.430791.

[62] Guangzhi Liao, Lizhi Xiao, and Ranhong Xie. "Method and experimental study of 2-D NMR logging". In: *Diffusion Fundamentals* 10 (2009). DOI: 10.62721/diffusion-fundamentals.10.448.

[63] David Linton Johnson and Lawrence M. Schwartz. "Analytic theory of two-dimensional NMR in systems with coupled macro- and micropores". In: *Physical Review E* 90.3 (2014). DOI: 10.1103/physreve.90.032407.

[64] Y.-Q. Song et al. "T1-T2 Correlation Spectra Obtained Using a Fast Two-Dimensional Laplace Inversion". In: *Journal of Magnetic Resonance* 154.2 (2002), 261-268. DOI: 10.1006/jmre.2001.2474.

[65] Marc Fleury and Maria Romero-Sarmiento. "Characterization of shales using T1-T2 NMR maps". In: *Journal of Petroleum Science and Engineering* 137 (2016), 55-62. DOI: 10.1016/j.petrol.2015.11.006.

[66] Zulkuf Azizoglu et al. "Simultaneous Assessment of Wettability and Water Saturation Through Integration of 2D NMR and Electrical Resistivity Measurements". In: *SPE Annual Technical Conference and Exhibition*. SPE, 2020. DOI: 10.2118/201519-ms.

[67] Mahmoud Elsayed et al. "A review on the applications of nuclear magnetic resonance (NMR) in the oil and gas industry: laboratory and field-scale measurements". In: *Journal of Petroleum Exploration and Production Technology* 12.10 (2022), 2747-2784. DOI: 10.1007/s13202-022-01476-3.

[68] Jonathan Mitchell and Edmund J. Fordham. "Contributed Review: Nuclear magnetic resonance core analysis at 0.3 T". In: *Review of Scientific Instruments* 85.11 (2014). DOI: 10.1063/1.4902093.

[69] Yi-Qiao Song and Ravinath Kausik. "NMR application in unconventional shale reservoirs - A new porous media research frontier". In: *Progress in Nuclear Magnetic Resonance Spectroscopy* 112-113 (2019), 17-33. DOI: 10.1016/j.pnmrs.2019.03.002.

[70] A. Timur. "Pulsed Nuclear Magnetic Resonance Studies of Porosity, Movable Fluid, and Permeability of Sandstones". In: *Journal of Petroleum Technology* 21.06 (1969), 775-786. DOI: 10.2118/2045-pa.

[71] Hiba M. Al-Janaee and Muwaqaf F. Al-shahwan. "Estimation of Porosity and Permeability by using Conventional Logs and NMR Log in Mishrif Formation/Buzurgan Oil Field". In: *Journal of Petroleum Research and Studies* 9.3 (2019), 75-89. DOI: 10.52716/jprs.v9i3.315.

[72] A. Mai and A. Kantzas. "Porosity Distributions in Carbonate Reservoirs Using Low-Field NMR". In: *Journal of Canadian Petroleum Technology* 46.07 (2007). DOI: 10.2118/07-07-02.

[73] David Walsh et al. "A Small-Diameter NMR Logging Tool for Groundwater Investigations". In: *Groundwater* 51.6 (2013), 914-926. DOI: 10.1111/gwat.12024.

[74] Richard F. Sigal. "Pore-Size Distributions for Organic-Shale-Reservoir Rocks From Nuclear-Magnetic-Resonance Spectra Combined With Adsorption Measurements". In: *SPE Journal* 20.04 (2015), 824-830. DOI: 10.2118/174546-pa.

[75] Razieh Solatpour and Apostolos Kantzas. "Application of nuclear magnetic resonance permeability models in tight reservoirs". In: *The Canadian Journal of Chemical Engineering* 97.5 (2019), 1191-1207. DOI: 10.1002/cjce.23354.

[76] Elijah Odusina, Carl Sondergeld, and Chandra Rai. "An NMR Study on Shale Wettability". In: *Canadian Unconventional Resources Conference*. SPE, 2011. DOI: 10.2118/147371-ms.

[77] Saurabh Tandon, Chelsea Newgord, and Zoya Heidari. "Wettability Quantification in Mixed-Wet Rocks Using a New NMR-Based Method". In: *SPE Reservoir Evaluation & Engineering* 23.03 (2020), 0896-0916. DOI: 10.2118/191509-pa.

[78] Andrea Valori, Farhan Ali, and Wael Abdallah. "Downhole Wettability: The Potential of NMR". In: *SPE EOR Conference at Oil and Gas West Asia*. SPE, 2018. DOI: 10.2118/190332-ms.

[79] Tianmin Jiang et al. "Evaluating Producible Hydrocarbons and Reservoir Quality in Organic Shale Reservoirs using Nuclear Magnetic Resonance (NMR) Factor Analysis". In: *SPE/CSUR Unconventional Resources Conference*. SPE, 2015. DOI: 10.2118/175893-ms.

[80] A. Tinni et al. "Nuclear-Magnetic-Resonance Response of Brine, Oil, and Methane in Organic-Rich Shales". In: *SPE Reservoir Evaluation & Engineering* 18.03 (2015), 400-406. DOI: 10.2118/168971-pa.

[81] R. Freedman, N. Heaton, and M. Flaum. "Field Applications of a New Nuclear Magnetic Resonance Fluid Characterization Method". In: *SPE Annual Technical Conference and Exhibition*. SPE, 2001. DOI: 10.2118/71713-ms.

[82] Gabor G. Hursan, James S. Deering, and Francis N. Kelly. "NMR Logs Help Formation Testing and Evaluation". In: *SPE Saudi Arabia Section Annual Technical Symposium and Exhibition*. SPE, 2015. DOI: 10.2118/177974-ms.

[83] Mauro Torres Ribeiro et al. "Integrated Petrophysics and Geosteering Reservoir Characterization in the Initial Development Phase of a Carbonate Reservoir - Campos Basin, Offshore Brazil". In: *OTC Brasil*. OTC, 2011. DOI: 10.4043/22738-ms.

[84] Hugh Daigle et al. "Nuclear magnetic resonance characterization of shallow marine sediments from the Nankai Trough, Integrated Ocean Drilling Program Expedition 333". In: *Journal of Geophysical Research: Solid Earth* 119.4 (2014), 2631-2650. DOI: 10.1002/2013JB010784.

[85] Son T. Dang, C. H. Sondergeld, and C. S. Rai. "Interpretation of Nuclear-Magnetic-Resonance Response to Hydrocarbons: Application to Miscible Enhanced-Oil-Recovery Experiments in Shales". In: *SPE Reservoir Evaluation & Engineering* 22.01 (2018), 302-309. DOI: 10.2118/191144-pa.

[86] RanHong Xie et al. "The influence factors of NMR logging porosity in complex fluid reservoir". In: *Science in China Series D: Earth Sciences* 51.S2 (2008), 212-217. DOI: 10.1007/s11430-008-6002-0.

[87] O. Mohnke et al. "Understanding NMR relaxometry of partially water-saturated rocks". In: (2014). DOI: 10.5194/hessd-11-12697-2014.

[88] Elliot Grunewald and Rosemary Knight. "A laboratory study of NMR relaxation times and pore coupling in heterogeneous media". In: *GEOPHYSICS* 74.6 (2009), E215-E221. DOI: 10.1190/1.3223712.

[89] Elliot Grunewald and Rosemary Knight. "A laboratory study of NMR relaxation times in unconsolidated heterogeneous sediments". In: *GEOPHYSICS* 76.4 (2011), G73-G83. DOI: 10.1190/1.3581094.

[90] Jonathan Mitchell et al. "A finite element approach to forward modeling of nuclear magnetic resonance measurements in coupled pore systems". In: *The Journal of Chemical Physics* 150.15 (2019). DOI: 10.1063/1.5092159.

[91] I. Hidajat et al. "Study of Vuggy Carbonates Using NMR and X-Ray CT Scanning". In: *SPE Reservoir Evaluation & Engineering* 7.05 (2004), 365-377. DOI: 10.2118/88995-pa.

[92] Huawei Zhao et al. "Applicability Comparison of Nuclear Magnetic Resonance and Mercury Injection Capillary Pressure in Characterisation Pore Structure of Tight Oil Reservoirs". In: *SPE Asia Pacific Unconventional Resources Conference and Exhibition*. SPE, 2015. DOI: 10.2118/177026-ms.

[93] FUYONG WANG, KUN YANG, and JIANCHAO CAI. "FRACTAL CHARACTERIZATION OF TIGHT OIL RESERVOIR PORE STRUCTURE USING NUCLEAR MAGNETIC RESONANCE AND MERCURY INTRUSION POROSIMETRY". In: *Fractals* 26.02 (2018), 1840017. DOI: 10.1142/s0218348x18400170.

[94] Erik Rylander et al. "NMR T2 Distributions in the Eagle Ford Shale: Reflections on Pore Size". In: *SPE Unconventional Resources Conference-USA*. SPE, 2013. DOI: 10.2118/164554-ms.

[95] Jibin ZHONG et al. "A decomposition method of nuclear magnetic resonance T2 spectrum for identifying fluid properties". In: *Petroleum Exploration and Development* 47.4 (2020), 740-752. DOI: 10.1016/s1876-3804(20)60089-1.

[96] Chaohui Lyu et al. "Application of NMR T2 to Pore Size Distribution and Movable Fluid Distribution in Tight Sandstones". In: *Energy & Fuels* 32.2 (2018), 1395-1405. DOI: 10.1021/acs.energyfuels.7b03431.

[97] Ziyue Li et al. "An NMR-based clay content evaluation method for tight oil reservoirs". In: *Journal of Geophysics and Engineering* 16.1 (2019), 116-124. DOI: 10.1093/jge/gyy010.

[98] R.L. Kleinberg and A. Boyd. "Tapered Cutoffs for Magnetic Resonance Bound Water Volume". In: *SPE Annual Technical Conference and Exhibition*. SPE, 1997. DOI: 10.2118/38737-ms.

[99] Quanpei Zhang et al. "Study of pore-throat structure characteristics and fluid mobility of Chang 7 tight sandstone reservoir in Jiyuan area, Ordos Basin". In: *Open Geosciences* 15.1 (2023). DOI: 10.1515/geo-2022-0534.

[100] Lu Chi, Kai Cheng, and Zoya Heidari. "Improved Assessment of Interconnected Porosity in Multiple-Porosity Rocks by Use of Nanoparticle Contrast Agents and Nuclear-Magnetic-Resonance Relaxation Measurements". In: *SPE Reservoir Evaluation & Engineering* 19.01 (2015), 095-107. DOI: 10.2118/170792-pa.

[101] Na Zhang et al. "Application of Multifractal Theory for Determination of Fluid Movability of Coal-Measure Sedimentary Rocks Using Nuclear Magnetic Resonance (NMR)". In: *Fractal and Fractional* 7.7 (2023), 503. DOI: 10.3390/fractalfract7070503.

[102] H. Pape et al. "Permeability Prediction for Low Porosity Rocks by Mobile NMR". In: *Pure and Applied Geophysics* 166.5-7 (2009), 1125-1163. DOI: 10.1007/s00024-009-0481-6.

[103] Keh-Jim Dunn, Gerald A. LaTorraca, and David J. Bergman. "Permeability relation with other petrophysical parameters for periodic porous media". In: *GEOPHYSICS* 64.2 (1999), 470-478. DOI: 10.1190/1.1444552.

[104] Katherine Dlubac, Rosemary Knight, and Kristina Keating. "A numerical study of the relationship between NMR relaxation and permeability in sands and gravels". In: *Near Surface Geophysics* 12.2 (2013), 219-230. DOI: 10.3997/1873-0604.2013042.

[105] Rosemary Knight et al. "NMR Logging to Estimate Hydraulic Conductivity in Unconsolidated Aquifers". In: *Groundwater* 54.1 (2015), 104-114. DOI: 10.1111/gwat.12324.

[106] Hui Gao and Huazhou Li. "Determination of movable fluid percentage and movable fluid porosity in ultra-low permeability sandstone using nuclear magnetic resonance (NMR) technique". In: *Journal of Petroleum Science and Engineering* 133 (2015), 258-267. DOI: 10.1016/j.petrol.2015.06.017.

[107] Katherine Dlubac et al. "Use of NMR logging to obtain estimates of hydraulic conductivity in the High Plains aquifer, Nebraska, USA". In: *Water Resources Research* 49.4 (2013), 1871-1886. DOI: 10.1002/wrcr.20151.

[108] Jing-Ya Zhang et al. "Microscopic oil occurrence in high-maturity lacustrine shales: Qingshankou Formation, Gulong Sag, Songliao Basin". In: *Petroleum Science* 20.5 (2023), 2726-2746. DOI: 10.1016/j.petsci.2023.08.026.

[109] Jonathan Mitchell et al. "Quantitative Remaining Oil Interpretation Using Magnetic Resonance: From the Laboratory to the Pilot". In: *SPE EOR Conference at Oil and Gas West Asia*. SPE, 2012. DOI: 10.2118/154704-ms.

[110] Jonathan Mitchell et al. "Quantitative In Situ Enhanced Oil Recovery Monitoring Using Nuclear Magnetic Resonance". In: *Transport in Porous Media* 94.3 (2012), 683-706. DOI: 10.1007/s11242-012-0019-8.

[111] Jonathan Mitchell et al. "Monitoring Chemical EOR Processes". In: *SPE Improved Oil Recovery Symposium*. SPE, 2014. DOI: 10.2118/169155-ms.

[112] Ahmad M. Al Harbi et al. "The Study of Nanosurfactant EOR in Carbonates by Advanced NMR Technique". In: *Abu Dhabi International Petroleum Exhibition & Conference*. SPE, 2017. DOI: 10.2118/188710-ms.

[113] Haitao Wang et al. "Nuclear Magnetic Resonance Study on Mechanisms of Oil Mobilization in Tight Reservoir Exposed to CO2 in Pore Scale". In: *SPE Improved Oil Recovery Conference*. SPE, 2016. DOI: 10.2118/179554-ms.

[114] Dao-Yi Zhu, Zhi-Hua Deng, and Si-Wei Chen. "A review of nuclear magnetic resonance (NMR) technology applied in the characterization of polymer gels for petroleum reservoir conformance control". In: *Petroleum Science* 18.6 (2021), 1760-1775. DOI: 10.1016/j.petsci.2021.09.008.

[115] J. Mitchell, A.M. Howe, and A. Clarke. "Real-time oil-saturation monitoring in rock cores with low-field NMR". In: *Journal of Magnetic Resonance* 256 (2015), 34-42. DOI: 10.1016/j.jmr.2015.04.011.

[116] Hyung Tae Kwak, Jinxun Wang, and Abdulkaim M. AlSofi. "Close Monitoring of Gel Based Conformance Control by NMR Techniques". In: *SPE Middle East Oil & Gas Show and Conference*. SPE, 2017. DOI: 10.2118/183719-ms.

[117] Ping Yang, Hekun Guo, and Daoyong Yang. "Determination of Residual Oil Distribution during Waterflooding in Tight Oil Formations with NMR Relaxometry Measurements". In: *Energy & Fuels* 27.10 (2013), 5750-5756. DOI: 10.1021/ef400631h.

[118] Shuang Liang et al. "Study on EOR method in offshore oilfield: Combination of polymer microspheres flooding and nitrogen foam flooding". In: *Journal of Petroleum Science and Engineering* 178 (2019), 629-639. DOI: 10.1016/j.petrol.2019.03.078.

[119] Yuechao Zhao et al. "CO2 flooding enhanced oil recovery evaluated using magnetic resonance imaging technique". In: *Energy* 203 (2020), 117878. DOI: 10.1016/j.energy.2020.117878.

[120] Baharak B. Alamdari, Mojtaba Kiani, and Hossein Kazemi. "Experimental and Numerical Simulation of Surfactant-Assisted Oil Recovery in Tight Fractured Carbonate Reservoir Cores". In: *SPE Improved Oil Recovery Symposium*. SPE, 2012. DOI: 10.2118/153902-ms.

[121] Ming Li et al. "Polymer Flooding Enhanced Oil Recovery Evaluated with Magnetic Resonance Imaging and Relaxation Time Measurements". In: *Energy & Fuels* 31.5 (2017), 4904-4914. DOI: 10.1021/acs.energyfuels.7b00030.

[122] Bin Zou et al. "Mechanisms of Permeability Alteration via Gel Based on Nuclear Magnetic Resonance". In: *Processes* 13.2 (2025), 497. DOI: 10.3390/pr13020497.

[123] Xing Huang et al. "Influence of Typical Core Minerals on Tight Oil Recovery during CO2 Flooding Using the Nuclear Magnetic Resonance Technique". In: *Energy & Fuels* 33.8 (2019), 7147-7154. DOI: 10.1021/acs.energyfuels.9b01220.

[124] Bo Ren et al. "Laboratory Assessment and Field Pilot of Near Miscible CO2 Injection for IOR and Storage in a Tight Oil Reservoir of ShengLi Oilfield China". In: *SPE Enhanced Oil Recovery Conference*. SPE, 2011. DOI: 10.2118/144108-ms.

[125] Peng Chen et al. "Experimental Investigation on CO2 Injection in Block M". In: *Journal of Chemistry* 2018 (2018), 1-7. DOI: 10.1155/2018/8623020.

[126] X. Dong et al. "Investigating Impacts of Fluid Types and Pore Sizes on Oil Recovery of N2-CO2 Huff-n-Puff with NMR". In: *82nd EAGE Annual Conference & Exhibition*. European Association of Geoscientists & Engineers, 2020, 1-5. DOI: 10.3997/2214-4609.202010154.

[127] K. Asghari et al. "Development of a Correlation Between Performance of CO2 Flooding and the Past Performance of Waterflooding in Weyburn Oil Field". In: *SPE/DOE Symposium on Improved Oil Recovery*. SPE, 2006. DOI: 10.2118/99789-ms.

[128] William S. Price. "Pulsed-field gradient nuclear magnetic resonance as a tool for studying translational diffusion: Part 1. Basic theory". In: *Concepts in Magnetic Resonance* 9.5 (1997), 299-336. DOI: 10.1002/(sici)1099-0534(1997)9:5<299::aid-cmr2>3.0.co;2-u.

[129] R.M. Cotts et al. "Pulsed field gradient stimulated echo methods for improved NMR diffusion measurements in heterogeneous systems". In: *Journal of Magnetic Resonance (1969)* 83.2 (1989), 252-266. DOI: 10.1016/0022-2364(89)90189-3.

[130] Mahmoud Elsayed et al. "New Technique for Evaluating Fracture Geometry and Preferential Orientation Using Pulsed Field Gradient Nuclear Magnetic Resonance". In: *SPE Journal* 26.05 (2021), 2880-2893. DOI: 10.2118/205505-pa.

[131] Gabor Hursan et al. "Oil Viscosity Estimation from NMR Logs for In-Situ Heavy Oil Characterization". In: *SPE Annual Technical Conference and Exhibition*. SPE, 2016. DOI: 10.2118/181600-ms.

[132] Bo Liu et al. "Investigation of oil and water migrations in lacustrine oil shales using 20 MHz 2D NMR relaxometry techniques". In: *Petroleum Science* 19.3 (2022), 1007-1018. DOI: 10.1016/j.petsci.2021.10.011.

[133] B.A. Baldwin and W.S. Yamanashi. "NMR imaging of fluid dynamics in reservoir core". In: *Magnetic Resonance Imaging* 6.5 (1988), 493-500. DOI: 10.1016/0730-725x(88)90123-3.

[134] J. Mitchell et al. "Magnetic resonance imaging in laboratory petrophysical core analysis". In: *Physics Reports* 526.3 (2013), 165-225. DOI: 10.1016/j.physrep.2013.01.003.

[135] M. P. Enwere and J. S. Archer. "NMR Imaging for Water/Oil Displacement in Cores Under Viscous-Capillary Force Control". In: *SPE/DOE Enhanced Oil Recovery Symposium*. SPE, 1992. DOI: 10.2118/24166-ms.

[136] B. Afsahi and A. Kantzas. "Advances in Diffusivity Measurement of Solvents in Oil Sands". In: *Canadian International Petroleum Conference*. PETSOC, 2006. DOI: 10.2118/2006-130.

[137] Mohamed Mehana and Ilham El-monier. "Shale characteristics impact on Nuclear Magnetic Resonance (NMR) fluid typing methods and correlations". In: *Petroleum* 2.2 (2016), 138-147. DOI: 10.1016/j.petlm.2016.02.002.

[138] I. Habina et al. "Insight into oil and gas-shales compounds signatures in low field 1H NMR and its application in porosity evaluation". In: *Microporous and Mesoporous Materials* 252 (2017), 37-49. DOI: 10.1016/j.micromeso.2017.05.054.

[139] Sun Hao et al. "Understanding Shale Gas Production Mechanisms Through Reservoir Simulation". In: *SPE/EAGE European Unconventional Resources Conference and Exhibition*. SPE, 2014. DOI: 10.2118/167753-ms.

[140] Utpalendu Kuila and Manika Prasad. "Specific surface area and pore-size distribution in clays and shales". In: *Geophysical Prospecting* 61.2 (2013), 341-362. DOI: 10.1111/1365-2478.12028.

[141] Scott H. Stevens, Keith D. Moodhe, and Vello A. Kuuskraa. "China Shale Gas and Shale Oil Resource Evaluation and Technical Challenges". In: *SPE Asia Pacific Oil and Gas Conference and Exhibition*. SPE, 2013. DOI: 10.2118/165832-ms.

[142] NIE Haikuan, ZHANG Jinchuan, and JIANG Shengling. "Types and Characteristics of the Lower Silurian Shale Gas Reservoirs in and Around the Sichuan Basin". In: *Acta Geologica Sinica - English Edition* 89.6 (2015), 1973-1985. DOI: 10.1111/1755-6724.12611.

[143] Zhanlei Wang et al. "Nanoscale Pore Characteristics of the Lower Permian Shanxi Formation Transitional Facies Shale, Eastern Ordos Basin, North China". In: *Frontiers in Earth Science* 10 (2022). DOI: 10.3389/feart.2022.842955.

[144] Gareth R. Chalmers, R. Marc Bustin, and Ian M. Power. "Characterization of gas shale pore systems by porosimetry, pycnometry, surface area, and field emission scanning electron microscopy/transmission electron microscopy image analyses: Examples from the Barnett, Woodford, Haynesville, Marcellus, and Doig units". In: *AAPG Bulletin* 96.6 (2012), 1099-1119. DOI: 10.1306/10171111052.

[145] Jin Lai et al. "Diagenesis and reservoir quality in tight gas sandstones: The fourth member of the Upper Triassic Xujiahe Formation, Central Sichuan Basin, Southwest China". In: *Geological Journal* 53.2 (2017), 629-646. DOI: 10.1002/gj.2917.

[146] Kaishuo Yang et al. "Shale rock core analysis using NMR: Effect of bitumen and water content". In: *Journal of Petroleum Science and Engineering* 195 (2020), 107847. DOI: 10.1016/j.petrol.2020.107847.

[147] F. Panattoni et al. "Improved Description of Organic Matter in Shales by Enhanced Solid Fraction Detection with Low-Field 1H NMR Relaxometry". In: *Energy & Fuels* 35.22 (2021), 18194-18209. DOI: 10.1021/acs.energyfuels.1c02386.

[148] Minglong Li et al. "Differences of Organic Matter Types of High Maturity Marine Organic-Rich Shale from Wufeng-Longmazi Formation: Implication for Shale Gas "Sweet Spots" Prediction". In: *Energy Exploration & Exploitation* 41.4 (2023), 1325-1343. DOI: 10.1177/01445987231171407.

[149] Xinhua Ma et al. "Insights into NMR response characteristics of shales and its application in shale gas reservoir evaluation". In: *Journal of Natural Gas Science and Engineering* 84 (2020), 103674. DOI: 10.1016/j.jngse.2020.103674.

[150] Seyedalireza Khatibi et al. "NMR relaxometry a new approach to detect geochemical properties of organic matter in tight shales". In: *Fuel* 235 (2019), 167-177. DOI: 10.1016/j.fuel.2018.07.100.

[151] Kurt Livo, Milad Saidian, and Manika Prasad. "Effect of paramagnetic mineral content and distribution on nuclear magnetic resonance surface relaxivity in organic-rich Niobrara and Haynesville shales". In: *Fuel* 269 (2020), 117417. DOI: 10.1016/j.fuel.2020.117417.

[152] Ismail Sulucarnain, Carl H. Sondergeld, and Chandra S. Rai. "An NMR Study of Shale Wettability and Effective Surface Relaxivity". In: *SPE Canadian Unconventional Resources Conference*. SPE, 2012. DOI: 10.2118/162236-ms.

[153] Jinbu Li et al. "Adsorbed and free hydrocarbons in unconventional shale reservoir: A new insight from NMR T1-T2 maps". In: *Marine and Petroleum Geology* 116 (2020), 104311. DOI: 10.1016/j.marpetgeo.2020.104311.

[154] Emilia V. Silletta et al. "Organic matter detection in shale reservoirs using a novel pulse sequence for T1-T2 relaxation maps at 2 MHz". In: *Fuel* 312 (2022), 122863. DOI: 10.1016/j.fuel.2021.122863.

[155] Yonghong FU et al. "Microscopic pore-fracture configuration and gas-filled mechanism of shale reservoirs in the western Chongqing area, Sichuan Basin, China". In: *Petroleum Exploration and Development* 48.5 (2021), 1063-1076. DOI: 10.1016/s1876-3804(21)60091-6.

[156] Donghan Yang and Ravinath Kausik. "23Na and 1H NMR Relaxometry of Shale at High Magnetic Field". In: *Energy & Fuels* 30.6 (2016), 4509-4519. DOI: 10.1021/acs.energyfuels.6b00130.

[157] C.R. Clarkson et al. "Nanopore-Structure Analysis and Permeability Predictions for a Tight Gas Siltstone Reservoir by Use of Low-Pressure Adsorption and Mercury-Intrusion Techniques". In: *SPE Reservoir Evaluation & Engineering* 15.06 (2012), 648-661. DOI: 10.2118/155537-pa.

[158] C.R. Clarkson et al. "Pore structure characterization of North American shale gas reservoirs using USANS/SANS, gas adsorption, and mercury intrusion". In: *Fuel* 103 (2013), 606-616. DOI: 10.1016/j.fuel.2012.06.119.

[159] Fu-Yong Wang, Kun Yang, and Yun Zai. "Multifractal characteristics of shale and tight sandstone pore structures with nitrogen adsorption and nuclear magnetic resonance". In: *Petroleum Science* 17.5 (2020), 1209-1220. DOI: 10.1007/s12182-020-00494-2.

[160] Artem Borysenko et al. "Experimental investigations of the wettability of clays and shales". In: *Journal of Geophysical Research: Solid Earth* 114.B7 (2009). DOI: 10.1029/2008jb005928.

[161] A. M. Serry, U. Herz, and L. Tagarieva. "Reservoir Characterization while Drilling; A Real Time Geosteering Answer to Maximize Well Values. A Case Study, Offshore Abu Dhabi". In: *Abu Dhabi International Petroleum Exhibition & Conference*. SPE, 2016. DOI: 10.2118/183092-ms.

[162] Wenjun Zhao et al. "Approaches of Combining Machine Learning with NMR-based Pore Structure Characterization for Reservoir Evaluation". In: (2023). DOI: 10.20944/preprints202312.0444.v1.

[163] L. DePavia et al. "A Next-Generation Wireline NMR Logging Tool". In: *SPE Annual Technical Conference and Exhibition*. SPE, 2003. DOI: 10.2118/84482-ms.

[164] Ridvan Akkurt et al. "Collaborative Development of a Slim LWD NMR Tool: From Concept to Field Testing". In: *SPE Saudi Arabia Section Technical Symposium*. SPE, 2009. DOI: 10.2118/126041-ms.

[165] Nicholas Heaton et al. "New Generation Magnetic Resonance While Drilling". In: *SPE Annual Technical Conference and Exhibition*. SPE, 2012. DOI: 10.2118/160022-ms.

[166] Jie Wang et al. "Theoretical investigation of heterogeneous wettability in porous media using NMR". In: *Scientific Reports* 8.1 (2018). DOI: 10.1038/s41598-018-31803-w.

[167] Klaas Nicolay et al. "Diffusion NMR spectroscopy". In: *NMR in Biomedicine* 14.2 (2001), 94-111. DOI: 10.1002/nbm.686.

[168] Clint P. Aichele et al. "Water in oil emulsion droplet size characterization using a pulsed field gradient with diffusion editing (PFG-DE) NMR technique". In: *Journal of Colloid and Interface Science* 315.2 (2007), 607-619. DOI: 10.1016/j.jcis.2007.07.057.

[169] Abubakar Isah et al. "Characterization of Fluid Drainage Mechanism at Core and Pore Scales: an NMR Capillary Pressure-Based Saturation Exponent Prediction". In: *SPE Europec featured at 82nd EAGE Conference and Exhibition*. SPE, 2021. DOI: 10.2118/205176-ms.

[170] Debora Salomon Marques et al. "Benchmarking of Pulsed Field Gradient Nuclear Magnetic Resonance as a Demulsifier Selection Tool with Arabian Light Crude Oils". In: *SPE Production & Operations* 36.02 (2020), 368-374. DOI: 10.2118/203820-pa.

[171] J M Salhany et al. "High resolution 31P nuclear magnetic resonance studies of intact yeast cells." In: *Proceedings of the National Academy of Sciences* 72.12 (1975), 4966-4970. DOI: 10.1073/pnas.72.12.4966.

[172] Gabor Hursan et al. "Study of OBM Invasion on NMR Logging - Mechanisms and Applications". In: *SPE Kingdom of Saudi Arabia Annual Technical Symposium and Exhibition*. SPE, 2018. DOI: 10.2118/192218-ms.

[173] Tetsuya Suekane et al. "Application of MRI in the Measurement of Two-Phase Flow of Supercritical CO2 and Water in Porous Rocks". In: *Journal of Porous Media* 12.2 (2009), 143-154. DOI: 10.1615/jpormedia.v12.i2.40.

[174] Yuechao Zhao et al. "Visualization of CO2 and oil immiscible and miscible flow processes in porous media using NMR micro-imaging". In: *Petroleum Science* 8.2 (2011), 183-193. DOI: 10.1007/s12182-011-0133-1.

[175] A Legchenko et al. "Three-dimensional magnetic resonance imaging for groundwater". In: *New Journal of Physics* 13.2 (2011), 025022. DOI: 10.1088/1367-2630/13/2/025022.

[176] J. Bryan et al. "In Situ Viscosity of Heavy Oil: Core and Log Calibrations". In: *Journal of Canadian Petroleum Technology* 46.11 (2007). DOI: 10.2118/07-11-04.

[177] Zhengshuai Liu et al. "Application of nuclear magnetic resonance (NMR) in coalbed methane and shale reservoirs: A review". In: *International Journal of Coal Geology* 218 (2020), 103261. DOI: 10.1016/j.coal.2019.103261.

[178] Majed F. Kanfar. "Real-Time Integrated Petrophysics: Geosteering in Challenging Geology & Fluid Systems". In: *SPE Saudi Arabia section Young Professionals Technical Symposium*. SPE, 2012. DOI: 10.2118/160922-ms.

[179] S. Tandon, A. Rostami, and Z. Heidari. "A New NMR-Based Method for Wettability Assessment in Mixed-Wet Rocks". In: *SPE Annual Technical Conference and Exhibition*. SPE, 2017. DOI: 10.2118/187373-ms.

[180] Chenglin Li et al. "Nuclear magnetic resonance pore radius transformation method and fluid mobility characterization of shale oil reservoirs". In: *Geoenergy Science and Engineering* 221 (2023), 211403. DOI: 10.1016/j.geoen.2022.211403.