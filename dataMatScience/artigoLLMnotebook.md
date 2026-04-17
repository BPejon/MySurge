# Applications of Nuclear Magnetic Resonance (NMR) in the Oil and Gas Industries

## 1. Introduction

Understanding the complex interactions between rocks and fluids in porous media is critical for both academic research and industrial operations in the oil and gas sector. Accurate determination of reservoir properties, such as pore connectivity and fluid saturation, is essential for assessing the value of hydrocarbon assets and designing effective completion strategies. While traditional logging tools like resistivity, density, and neutron logs provide useful data, they often respond to the rock matrix or framework rather than the fluids alone, making interpretation difficult in complex lithologies. **Nuclear Magnetic Resonance (NMR) has emerged as a superior alternative because it provides direct, non-destructive measurements of the hydrogen nuclei (protons) contained within reservoir fluids.** By probing these protons, NMR offers a "window" into the pore space, enabling precise characterization of porosity, permeability, and fluid types that conventional methods cannot match.

### 1.1 NMR as a Paradigm Shift: From Lab to Borehole

The evolution of NMR in the petroleum industry represents a significant technical leap from its roots in laboratory molecular structure determination and medical diagnostics. Originally discovered in 1946, the technology was commercialized for oilfield use in the early 1990s. A fundamental shift occurred with the development of **"inside-out" NMR logging tools**; unlike laboratory spectrometers or medical MRI machines where the subject is placed inside the magnet, borehole NMR tools are placed inside the wellbore to analyze the surrounding formation. This transition required overcoming massive engineering challenges, including inhomogeneous magnetic fields and extreme downhole conditions of temperature and pressure. Today, the technology has matured from a niche laboratory experiment into a primary formation evaluation tool used in both wireline and logging-while-drilling (LWD) environments.

### 1.2 Core Advantages: Matrix Independence and Direct Fluid Sensing

The primary advantage of NMR over conventional logging tools is its **lithology-independent measurement of porosity [1.2, 496, 1161].** Traditional porosity tools, such as neutron and bulk-density logs, are influenced by the rock framework and mineralogy, requiring complex calibrations for different lithologies. In contrast, because NMR signals only originate from the fluids (water, oil, and gas) and the solids produce signals that decay too fast to be recorded, the resulting porosity measurement is independent of the rock matrix. Furthermore, NMR is uniquely capable of providing three distinct types of information simultaneously:

- **Fluid Quantity:** Direct measurement of hydrogen density to determine apparent porosity.
- **Fluid Properties:** Using relaxation times ($T_1$ and $T_2$) and diffusivity ($D$) to distinguish between bound water, movable water, gas, and oils of varying viscosity.
- **Pore Geometry:** Relating relaxation spectra to pore size distributions, which is critical for estimating formation permeability.

### 1.3 Scope of the Review: Multiscale Characterization and Future Trends

This review examines the comprehensive application of NMR across multiple scales, ranging from nanometer-scale laboratory core analysis to decameter-scale field logging. It highlights how **low-field NMR (typically operating at ~2 MHz)** serves as a bridge, allowing for the cross-validation of laboratory findings with in-situ field data. The review covers advanced petrophysical characterization, including wettability assessment and damage evaluation from drilling mud invasion. Special emphasis is placed on the characterization of **unconventional reservoirs**, such as shale and tight gas, where NMR is essential for identifying organic vs. inorganic porosity. Finally, the scope extends to emerging developments in the "Digital Oilfield," including the integration of **Artificial Intelligence and Machine Learning** for automated data inversion, as well as the strategic role of NMR in the **energy transition**, specifically in carbon capture and hydrogen storage.

## 2. Fundamentals of NMR Physics and Relaxation Mechanisms

Nuclear magnetic resonance (NMR) is a quantum mechanical phenomenon based on the interaction between the magnetic moments of specific atomic nuclei and an external magnetic field. In the oil and gas industry, the hydrogen nucleus ($^1H$), or proton, is the primary target of investigation due to its high natural abundance and large gyromagnetic ratio.

### 2.1 Nuclear Magnetism and the Larmor Condition

Atomic nuclei with a non-zero spin quantum number ($I > 0$) possess an intrinsic magnetic moment. In the absence of an external field, these moments are randomly oriented. When placed in a static magnetic field ($B_0$), the nuclei align and precess around the field direction, a state described as polarization. The frequency of this precession, known as the **Larmor frequency ($\omega$)**, is directly proportional to the strength of the static field:

$$ \omega = \gamma B_0 $$

where $\gamma$ is the gyromagnetic ratio, a constant unique to each nucleus. Table 1 summarizes typical values for relevant nuclei.

**Table 1: Typical values for some nuclei**

| Nucleus | Spin Number ($I$) | $\gamma$ ($10^6$ rad $s^{-1}T^{-1}$) | Natural Abundance (%) |
| :--- | :---: | :---: | :---: |
| $^1H$ | 1/2 | 267.522 | $\sim$99.9 |
| $^{13}C$ | 1/2 | 67.283 | 1.1 |
| $^{23}Na$ | 3/2 | 70.761 | 100 |

The net magnetization ($M_0$) generated by this alignment is proportional to the number of protons and the strength of $B_0$, while inversely proportional to the absolute temperature. To detect a signal, an oscillating radio-frequency (RF) magnetic field ($B_1$) is applied perpendicular to $B_0$ at the Larmor frequency to "tip" the magnetization into the transverse plane.

### 2.2 Relaxation Mechanisms in Porous Media

Once the $B_1$ field is removed, the nuclear spins return to their equilibrium state through two distinct processes: **Longitudinal Relaxation ($T_1$)**, the recovery of magnetization along the $B_0$ axis, and **Transverse Relaxation ($T_2$)**, the loss of phase coherence in the transverse plane. In reservoir rocks, the relaxation of pore fluids is dominated by three simultaneous, parallel mechanisms:

$$ \frac{1}{T_{1,2}} = \frac{1}{T_{1,2,\text{bulk}}} + \frac{1}{T_{1,2,\text{surface}}} + \frac{1}{T_{1,2,\text{diffusion}}} $$

- **Bulk Relaxation ($T_{\text{bulk}}$):** This is the intrinsic relaxation of the fluid itself, controlled by its molecular composition and viscosity. Generally, $T_{\text{bulk}}$ is inversely related to viscosity, meaning viscous oils relax much faster than light hydrocarbons or water.
- **Surface Relaxation ($T_{\text{surface}}$):** This occurs at the fluid-solid interface due to interactions between the fluid protons and paramagnetic ions (e.g., $Fe^{3+}$, $Mn^{2+}$) on the pore walls. It is governed by the surface-to-volume ratio ($S/V$) of the pore and the surface relaxivity ($\rho$):
    $$ \frac{1}{T_{1,2,\text{surface}}} \approx \rho_{1,2} \left(\frac{S}{V}\right) $$
    Because smaller pores have a higher $S/V$ ratio, they exhibit shorter relaxation times, making $T_2$ a direct proxy for pore size.
- **Diffusion-Induced Relaxation ($T_{\text{diffusion}}$):** This mechanism only affects $T_2$ and occurs when protons diffuse through magnetic field gradients ($G$). The dephasing caused by this motion is expressed as:
    $$ \frac{1}{T_{2,\text{diffusion}}} = \frac{D(\gamma G T_E)^2}{12} $$
    where $D$ is the molecular diffusion coefficient and $T_E$ is the inter-echo spacing. High-diffusivity fluids like gas exhibit a strong diffusion-induced shift toward shorter $T_2$ values compared to oil or water.

### 2.3 Mathematical Descriptions

The dynamics of magnetization are classically described by the **Bloch equations**. For a single pore containing a wetting phase, the $T_1$ recovery and $T_2$ decay are often simplified to single exponential forms. However, reservoir rocks contain a distribution of pore sizes and fluid types, leading to a **multi-exponential decay** of the total magnetization:

$$ M(t) = \sum_{i=1}^{m} \phi_i \exp\left(-\frac{t}{T_{2,i}}\right) $$

where $\phi_i$ is the porosity component associated with the relaxation time $T_{2,i}$. The acquired raw signal (echo train) is transformed into a $T_2$ distribution through a mathematical inversion process, typically a Laplace Transform of the first kind. This distribution provides the foundational data for petrophysical analysis, where the area under the curve is calibrated to total porosity.

## 3. Advanced NMR Techniques: Pulse Sequences and Data Processing

While basic NMR physics provides the foundation, advanced pulse sequences and sophisticated data processing are what enable the practical separation of fluid types and the mapping of complex pore geometries in reservoir rocks.

### 3.1 1D Pulse Sequences: CPMG and Inversion Recovery

The two fundamental 1D measurements in the oil and gas industry are the transverse relaxation time ($T_2$) and the longitudinal relaxation time ($T_1$).

- **CPMG for $T_2$:** The Carr-Purcell-Meiboom-Gill (CPMG) sequence is the workhorse of oilfield NMR. It begins with a 90° radio-frequency (RF) pulse to tip the magnetization into the transverse plane, followed by a series of 180° refocusing pulses to generate an "echo train". This sequence minimizes the dephasing effects of static magnetic field inhomogeneities. The decay of the echo train is described as:
    $$ \frac{M_{xy}(nt_e)}{M_0} = \exp\left(-\frac{nt_e}{T_2}\right) $$
    where $t_e$ is the inter-echo spacing and $n$ is the number of echoes.
- **Inversion Recovery (IR) for $T_1$:** To measure $T_1$, an Inversion Recovery sequence is typically used. A 180° pulse inverts the magnetization, which is then allowed to recover for a variable delay time ($\tau_1$) before being tipped into the detection plane by a 90° pulse. The recovery curve follows:
    $$ \frac{M_z(\tau_1)}{M_0} = 1 - 2\exp\left(-\frac{\tau_1}{T_1}\right) $$
    In reservoir environments, $T_1$ is critical for distinguishing light hydrocarbons (which have long $T_1$) from water (which has shorter $T_1$ in rock pores).

### 3.2 2D and 3D Correlation Maps

Modern formation evaluation relies on multi-dimensional NMR to resolve overlapping signals in 1D spectra.

- **$T_1-T_2$ Correlation:** By varying the wait time ($TW$) and measuring the resulting $T_2$ decay, a 2D map is generated. This is particularly useful in shales and unconventional reservoirs to identify organic matter (high $T_1/T_2$ ratio) versus inorganic water (low $T_1/T_2$ ratio).
- **$D-T_2$ (Diffusion-Relaxation) Maps:** By implementing pulse field gradients (PFG) to measure the diffusion coefficient ($D$), log analysts can separate fluids based on their molecular mobility. Gas, possessing high diffusivity, shifts significantly on these maps compared to oil or water.
- **3D Fluid Characterization (3DFC):** Advanced logging tools now employ 3DFC, which uses multiple polarization times and varying inter-echo spacings ($T_e$) to simultaneously acquire $T_1$, $T_2$, and $D$ data. This reduces uncertainty in complex heavy-oil or mixed-saturation environments.

### 3.3 Data Inversion and Pre-treatment

Transforming raw echo data into petrophysical parameters is an "ill-posed" mathematical problem, meaning infinite solutions can exist in the presence of noise.

- **Inverse Laplace Transform (ILT):** The decay signal is a multi-exponential sum that must be inverted to find the distribution of relaxation times. This is typically formulated as:
    $$ M = KF + e $$
    where $M$ is the acquired data, $K$ is the kernel matrix, $F$ is the target distribution, and $e$ is the noise.
- **Regularization:** Tikhonov regularization is commonly applied to stabilize the solution by adding a penalty term to the minimization function. The smoothing parameter ($\alpha$) is often optimized using methods like the L-curve or Generalized Cross-Validation (GCV).
- **Pre-treatment Denoising:** Recent advancements have introduced Machine Learning (ML) for denoising raw echoes. Unsupervised methods like Double Sparsity Dictionary Learning (DSDL) can adaptively extract signals from noisy low-field data, significantly improving the resolution of the final $T_2$ spectrum.

## 4. Laboratory Petrophysics and Reservoir Characterization

Laboratory NMR measurements on core plugs serve as the "gold standard" for calibrating borehole logs and providing detailed petrophysical insights. Unlike macroscopic tests, NMR provides a pore-scale "window" into the fluid distribution and rock-fluid interactions.

### 4.1 Porosity Partitioning

One of the most powerful applications of NMR is the ability to partition total porosity into different fluid compartments based on their mobility. As shown in the sources, the total NMR porosity ($\phi_{\text{total}}$ or MSIG) is subdivided into components representing various levels of confinement:

- **Clay-Bound Water (CBW/MCBW):** Water ionically attached to clay surfaces, exhibiting extremely short $T_2$ times (typically $< 3.3$ ms).
- **Effective Porosity (MPHI):** The remaining porosity available for hydrocarbons and movable water.
- **Capillary-Bound Water (BVI):** Water held in small pores or as films by capillary forces, which is immobile under normal production pressures.
- **Free Fluid Index (FFI/MFFI):** The volume of movable fluids (water, oil, or gas).

A summary of the partitioning logic is provided in Table 2.

**Table 2: NMR Porosity Partitioning Terminology**

| Component | Symbol | Definition |
| :--- | :---: | :--- |
| Total Porosity | $\phi_{nmr}$ (MSIG) | Total hydrogen content (CBW + Effective) |
| Effective Porosity | $\phi_{eff}$ (MPHI) | Connected pore volume (BVI + FFI) |
| Bound Fluid Vol. | BVI (BFV) | Immobile capillary-bound water |
| Free Fluid Index | FFI (MFFI) | Movable hydrocarbons and water |

### 4.2 Pore Size Distribution (PSD) and MICP Correlation

The $T_2$ distribution is a direct proxy for the pore size distribution in water-saturated rocks because the relaxation rate is dominated by the surface mechanism. The relationship is governed by:

$$ \frac{1}{T_{2,\text{surface}}} = \rho_2 \left(\frac{S}{V}\right)_{\text{pore}} $$

Laboratory studies use this principle to relate $T_2$ spectra to **Mercury Injection Capillary Pressure (MICP)** data. While NMR measures the **pore body size**, MICP is sensitive to **pore throat size**. In many sandstones, a consistent scaling factor (the surface relaxivity, $\rho$) allows the two distributions to nearly overlay, providing a robust model for characterizing reservoir storage and flow capacity.

### 4.3 Permeability Modeling

NMR offers a unique way to estimate permeability without flow tests by using the relationship between porosity and pore geometry. The two primary models used in the industry are:

- **SDR (Schlumberger-Doll Research) Model:** Uses the geometric mean of the $T_2$ distribution ($T_{2gm}$) to represent the "average" pore size:
    $$ k_{\text{SDR}} = a \phi^4 T_{2gm}^2 $$
- **Timur-Coates (Free-Fluid) Model:** Relates permeability to the ratio of movable to bound fluids:
    $$ k_{\text{Coates}} = \left[ \frac{\phi}{C} \right]^4 \left( \frac{FFI}{BVI} \right)^2 $$
    where $a, C$, and the exponents are constants typically determined through core-log integration.

### 4.4 Wettability Assessment

Wettability describes the preference of a rock surface to be in contact with a specific fluid. In laboratory NMR, wetting fluids exhibit significantly shorter relaxation times because they are in direct contact with the mineral surfaces, whereas non-wetting fluids relax at their bulk rates. Recent laboratory advancements have introduced the **NMR Wettability Index (NWI)**, which compares the $T_2$ distributions of fluids in a "native" state versus fully saturated states to quantify whether a reservoir is water-wet, oil-wet, or mixed-wet. This is critical for predicting recovery efficiency in **Enhanced Oil Recovery (EOR)** operations.

### 4.5 NMR Cryoporometry

For tight reservoirs and shales with nanopores, standard relaxometry can be challenging. **NMR Cryoporometry** is used to characterize these systems by measuring the depression of the melting point of a liquid (like water or cyclohexane) confined within the pores. The melting point depression ($\Delta T$) is inversely proportional to the pore radius ($x$), following the **Gibbs-Thomson equation**:
$$ \Delta T = T_0 - T(x) = \frac{\mathbf{k}}{x} $$
where $\mathbf{k}$ is a constant related to the thermodynamic properties of the liquid and the surface energy of the pore wall. This technique provides high-resolution PSDs for pores in the 2 nm to 1 $\mu$m range.

## 5. Unconventional Reservoir Characterization (Shale and Tight Gas)

Characterizing unconventional reservoirs, particularly organic-rich shales, presents a unique challenge due to their multi-scale pore systems and the presence of complex organic matter. NMR is instrumental in these environments because it can distinguish between fluid types and provide insights into the geochemical state of the rock matrix.

### 5.1 Organic Matter Analysis

A primary goal in shale evaluation is distinguishing between the different organic components: solid kerogen, viscous bitumen, and light hydrocarbons. Standard CPMG sequences often fail to capture solid kerogen because its relaxation is too rapid. To address this, specialized techniques like **Gaussian-Exponential Inversion** are used to capture the extremely short $T_2^*$ signals from semi-solids.

The magnetization decay in these rapidly relaxing systems is modeled as a combination of Gaussian and multi-exponential components:

$$ M_{xy}(t) = \sum A_i \exp\left(-\left\{\frac{t}{T_{2i}^*}\right\}^2\right) + \sum B_i \exp\left(-\frac{t}{T_{2i}}\right) $$

where:
- $A_i$ represents the amplitude of the **Gaussian decay** from solid/semi-solid components (kerogen/bitumen).
- $B_i$ represents the amplitude of the **exponential decay** from mobile fluids (oil/water).
- $T_{2i}^*$ is the characteristic time constant for the Gaussian dephasing.

### 5.2 Maturity and Geochemistry

NMR serves as a proxy for the thermal maturity of shale by monitoring changes in the kerogen structure. As kerogen matures, it undergoes a chemical shift from aliphatic (chain-like) structures to **aromatic (ring-like) structures**, which is detectable via $^{13}C$ solid-state NMR spectroscopy.

Additionally, the **$T_1/T_2$ ratio** is a critical indicator of maturity and fluid type in shales. High $T_1/T_2$ ratios (often $> 10$) are typically associated with organic matter and viscous bitumen, while inorganic water typically exhibits lower ratios closer to unity. Multi-dimensional $T_1-T_2$ maps are therefore essential for identifying the "organic" signal signature amidst complex mineralogy.

### 5.3 Stress-Dependent Permeability

Shale reservoirs are highly sensitive to confining pressure, which significantly impacts their storage and flow capacity. Recent research has utilized NMR to quantify the **stress sensitivity** of different pore categories. The pore system is often divided into:

- **Seepage Pores (Soft Part):** Larger pores and micro-fractures that are highly compressible and dominate initial flow.
- **Adsorption Pores (Hard Part):** Nanometer-scale pores within the organic matrix that are more resistant to stress.

As confining pressure increases, NMR $T_2$ spectra shift toward shorter times, reflecting pore shrinkage. By monitoring these shifts under varying stress, researchers can develop heterogeneous compressibility models to predict permeability reduction during reservoir drawdown.

## 6. Field-Scale Applications: Wireline and LWD Logging

The transition of NMR technology from the laboratory to the wellbore has revolutionized formation evaluation by providing real-time, lithology-independent data on reservoir storage and flow capacity. Field applications are primarily conducted via two platforms: **Wireline Logging**, typically performed after drilling, and **Logging While Drilling (LWD)**, which provides critical data during the drilling process.

### 6.1 Logging While Drilling (LWD-NMR)

LWD-NMR enables real-time formation evaluation, which is instrumental for **geosteering** and optimizing well placement within the most productive zones ("sweet spots").

- **Operational Advantage:** By acquiring data during drilling, petrophysical information is obtained before significant mud filtrate invasion occurs, reducing borehole risk and allowing for timely completion decisions.
- **Geosteering Example:** In the Meji field offshore Nigeria, LWD-NMR was successfully used to drill a 1,500 ft lateral section through the most productive part of a thin sand by identifying high-permeability zones in real-time.
- **Motion Challenges:** Unlike wireline, LWD tools face extreme drilling dynamics, including fast rotation and lateral vibration. Lateral motion correction is essential for $T_2$ measurements, as dephasing caused by tool movement can lead to signal loss and vertical resolution degradation.

### 6.2 Tool Design Innovations

Modern borehole NMR tools have evolved into complex "inside-out" sensors designed to investigate cylindrical shells of the formation.

- **Magnet Configurations:**
    - **Centric-type:** These tools feature a larger, rotationally symmetric sensitive volume all around the borehole, providing better $B_1$ uniformity.
    - **Side-looking (Pad-type):** These are pressed against the borehole wall to minimize power requirements and reception sensitivity while targeting specific azimuthal regions.
    - **Saddle-Point vs. Gradient:** Saddle-point tools (e.g., Jackson design) target a region of near-zero magnetic field gradient to minimize motion sensitivity, though they are more prone to temperature-induced signal shifts. Gradient tools utilize multiple **Depths of Investigation (DOI)** to look past the mudcake into the virgin formation.
- **Multi-frequency Operation:** Advanced tools like the MRIL-Prime utilize up to **nine frequencies** to acquire data from nine closely spaced cylindrical volumes. This dramatically increases measurement efficiency, often allowing total porosity, $T_1$, $T_2$, and fluid typing to be captured in a single log pass.
- **Vertical Resolution Modeling:** The vertical resolution ($VR$) of a logging tool is a function of antenna length ($L$), logging speed ($V$), and acquisition parameters:
    $$ VR = L + V(TC \cdot RA - TW) $$
    where $TC$ is the cycle time, $RA$ is the running average, and $TW$ is the wait time.

### 6.3 Special Logging Challenges

Applying NMR in the field requires addressing specific environmental and reservoir complexities.

- **Tar and Heavy Oil Detection:** NMR is uniquely suited for real-time tar detection, which is critical because tar acts as a permeability barrier and is often unrecoverable. In carbonates, NMR-LWD integrates triple-combo data to flag low-mobility zones and adjust well trajectories.
- **Thin-Bed Characterization:** New high-resolution sensors (e.g., XMR) utilize shorter antennas and stronger magnetic gradients to reduce vertical averaging requirements, enabling the evaluation of thin laminations in unconventional reservoirs that conventional tools might miss.
- **Environmental Corrections:** While NMR is essentially matrix-independent, factors like **borehole rugosity**, conductive drilling muds, and temperature variation must be managed. The thickness of the investigated sensitive volume shell ($\Delta r$) is determined by the RF pulse bandwidth ($\Delta f$) and the field gradient ($G$):
    $$ \Delta r = \frac{\Delta f}{\gamma G} $$
    Precise control of these parameters ensures the tool investigates the formation beyond the mudcake layer.

## 7. Production Optimization and Flow Assurance

Nuclear magnetic resonance has expanded beyond formation evaluation into production engineering, where it serves as a non-invasive tool for monitoring fluid flow and ensuring the efficient transport of hydrocarbons from the reservoir to the surface.

### 7.1 Multiphase Flow Metering

Accurate measurement of oil, water, and gas flow rates is essential for reservoir management and production optimization. Traditional multiphase flow meters (MPFMs) often rely on radioactive sources (e.g., Gamma densitometers), which pose safety and regulatory challenges. **Magnetic resonance technology offers a non-radioactive alternative by measuring the nuclear magnetic properties of the flowing fluid stream.**

- **Operating Principle:** MR flow meters utilize low-field MRI to characterize flow regimes and quantify liquid phases. The tool measures longitudinal ($T_1$) and transverse ($T_2$) relaxation times, which are sensitive to fluid composition and flow velocity.
- **Fluid Characterization:** Because the $T_2$ of oil and water varies with temperature and well conditions, the flowmeter software stores multiple sets of fluid characteristics to ensure accurate calculations across different wells connected to the same manifold.
- **Accuracy:** Sensitivity analysis indicates that an 8% deviation in relaxation time measurements results in approximately a 4% error in gas and liquid flow rates, which is within the acceptable precision limits for industrial flow metering.

### 7.2 Enhanced Oil Recovery (EOR) Monitoring

NMR provides a unique "pore-scale window" into displacement mechanisms, making it superior to macroscopic techniques that only assess bulk recovery. It allows researchers to monitor how different EOR agents—such as CO₂, surfactants, or polymers—interact with specific pore systems.

- **Displacement Mechanisms:** NMR can define the recovery of oil and gas from different pore sizes, showing, for example, which fluids are recovered from small versus medium pores during CO₂ injection cycles.
- **Saturation Profiling:** NMR logging and laboratory coreflooding integration provide continuous profiles of oil saturation and monitor the progress of injected chemicals. This is particularly useful for evaluating the effectiveness of chemical EOR pilots in real-time.
- **Pore Structure Alteration:** Studies on "Huff-'n'-Puff" EOR in shales utilize NMR to assess changes in rock microstructure and fluid mobility resulting from cyclic gas injection.

### 7.3 Emulsion Characterization

Flow assurance is often challenged by the formation of stable emulsions (dispersions of oil in water or water in oil) during production and EOR operations. **Pulsed Field Gradient (PFG) NMR is the industry standard for non-destructively determining emulsion droplet size distributions.**

- **Droplet Sizing:** PFG-NMR with diffusion editing (PFG-DE) exploits the restricted diffusion of protons within droplets to calculate their size. This is governed by the relationship:
    $$ \frac{S(g)}{S_0} = \exp\left(-D(\gamma\delta g)^2 \Delta\right) $$
    where $g$ is the gradient strength and $D$ is the self-diffusion coefficient. Signal attenuation is plotted to obtain a slope representing the restricted diffusion within the droplets.
- **Demulsifier Selection:** NMR serves as a benchmark tool for selecting the most effective demulsifiers for specific crude oils by monitoring the breakage mechanism and coalescence rate of emulsions over time.
- **Stability Monitoring:** By quantifying the relative hydrogen index (RHI) and $T_2$ shifts, NMR identifies the impact of salinity and surfactant concentration on the long-term stability of produced emulsions.

---

## 8. Digital Oilfield: Integration of AI and Machine Learning

The integration of Artificial Intelligence (AI) and Machine Learning (ML) into NMR workflows represents a modern shift toward automated, high-accuracy formation evaluation. These techniques address the "ill-posed" nature of NMR inversion and the low signal-to-noise ratio (SNR) characteristic of low-field measurements.

### 8.1 Data-Driven Inversion and SNR Enhancement

Traditional NMR denoising often relies on fixed mathematical transformations (e.g., wavelets) or hand-designed feature selectors, which are frequently non-adaptive. Modern data-driven approaches utilize **Dictionary Learning** and **Deep Learning** to adaptively extract signals from noisy data.

- **Dictionary Learning:** This machine learning method is based on sparse representation theory. Available information in noisy NMR echo data can be adaptively reconstructed, significantly improving the accuracy of the $T_2$ spectrum even at high noise levels.
- **Discrete Cosine Transform (DCT):** Novel denoising methods using variable-length windows and DCT have been proposed to improve the reliability of inverted spectra. These methods are particularly effective in resolving overlapping fluid signals on multi-dimensional NMR maps that would otherwise be obscured by noise.
- **SNR Improvements:** By pre-treating raw echoes with ML frameworks, researchers have observed a substantial enhancement in the precision of apparent porosity and $T_2$ distribution estimates compared to traditional Tikhonov regularization alone.

### 8.2 Automated Fluid Identification

Machine learning models are increasingly used to classify hydrocarbon phases by processing multi-dimensional NMR data (such as $T_1-T_2$ and $D-T_2$ maps).

- **Workflow Integration:** ML workflows for nuclear data analysis typically include preprocessing, model training, validation, and uncertainty quantification. In the context of NMR, these models can be trained on large "Digital Rock Physics" datasets to recognize specific "fingerprints" of oil, gas, and water.
- **Efficiency:** Automated classification reduces the bias inherent in manual log analysis and allows for real-time fluid typing during LWD operations, where rapid decision-making is critical.

### 8.3 Physics-Informed Neural Networks (PINNs)

**Physics-Informed Neural Networks (PINNs)** represent a sophisticated frontier where physical laws—such as the Bloch equations or relaxation mechanisms—are incorporated directly into the neural network's loss function.

- **Mathematical Constraints:** Instead of acting as a "black box," the network is constrained by the physics of the NMR experiment:
    $$ X_{DT}(t) = f(X_{\text{real}}(t), u(t)) $$
    where $X_{DT}$ denotes the "Digital Twin" or virtual replica state vector, $X_{\text{real}}$ represents the real-time sensor data, and $u(t)$ is the control input.
- **Predictive Capabilities:** PINNs can simulate complex reactor or reservoir dynamics, allowing operators to test "what-if" scenarios in a safe, virtual environment before applying them to physical wells. This integration ensures that the AI's output remains physically consistent with known petrophysical principles, leading to more stable and trustworthy results in heterogeneous formations.

## 9. NMR in the Energy Transition

As the global energy landscape shifts, nuclear magnetic resonance technology is being repurposed from traditional hydrocarbon evaluation to support low-carbon initiatives. The sources indicate that NMR is uniquely capable of characterizing the storage potential and transport mechanisms of gases critical to the energy transition.

### 9.1 Carbon Capture and Geo-Sequestration (CCS)

Nuclear magnetic resonance serves as a "pore-scale window" into the displacement mechanisms during CO$_2$ injection. It is increasingly used to **investigate the geostorage of CO$_2$** by quantifying how it interacts with different pore sizes in reservoir rocks.

- **Monitoring Displacement:** NMR can identify which fluids are recovered from small versus medium pores during CO$_2$ flooding cycles.
- **Saturation Profiling:** NMR logging and laboratory coreflooding provide distribution profiles for chemical and oil saturations, which is critical for evaluating the efficiency of sequestration pilots in real-time.
- **Geochemical Interactions:** Multi-dimensional NMR (e.g., $T_1-T_2$ maps) can potentially monitor pore-scale CO$_2$-brine-rock interactions, which affect long-term storage security.

### 9.2 Underground Hydrogen Storage (UHS)

Hydrogen is a critical carrier for renewable energy, and its storage in subsurface reservoirs (shales, salt caverns, or depleted aquifers) is a major research frontier. NMR is instrumental in investigating the **geostorage of H$_2$.**

- **Isotopic Sensitivity:** NMR is sensitive to various hydrogen isotopes. While $^1H$ is the standard target, isotopes like $^2H$ (deuterium) can also be tracked to investigate molecular transport and solubility (see Table 3).
- **Transport in Nanopores:** In unconventional shales, NMR identifies "organic matter pore signals" and can distinguish between mobile H$_2$ in large fractures and adsorbed hydrogen in nanopores.

**Table 3: Typical Values for Nuclei Relevant to Storage Studies**

| Nucleus | Spin Number ($I$) | $\gamma$ ($10^6$ rad $s^{-1}T^{-1}$) | Natural Abundance (%) |
| :--- | :---: | :---: | :---: |
| $^1H$ | 1/2 | 267.522 | $\sim$99.9 |
| $^2H$ | 1 | 41.066 | 0.015 |
| $^{23}Na$ | 3/2 | 70.761 | 100 |

(Source:)

### 9.3 Geothermal and High-Temperature Reservoir Characterization

Geothermal energy production requires formation evaluation in extreme environments. NMR tools are designed to operate and provide high-accuracy data even as temperatures rise, although **high temperatures significantly impact NMR physics.**

- **B$_0$ Stability:** As the permanent magnet in a logging tool becomes hotter, the static magnetic field ($B_0$) decreases, which in turn **decreases the depth of investigation (DOI).**
- **Temperature Compensation:** For accurate evaluation in hot geothermal reservoirs, the $B_1$ pulse strength must be corrected for borehole temperature using internal sensor data (Temp1, Temp2, and Temp3).
    $$ B_{1\text{mod}} = B_1 [1 + 0.0033(T_{\text{magnet}} - T_{\text{calibration}})] $$
- **Fluid Properties:** Viscosity is inversely related to temperature; therefore, in high-temperature reservoirs, relaxation times ($T_1$ and $T_2$) are shifted toward longer values, requiring careful adjustment of logging wait times ($TW$).

## 10. Strategic Synthesis and Future Outlook

The final section of the review synthesizes the current state of NMR technology and looks forward to its role in the evolving energy landscape. While NMR has moved from a laboratory curiosity to a cornerstone of formation evaluation, several technical and operational hurdles remain, alongside promising new avenues for research and field implementation.

### 10.1 Persistent Challenges

Despite its advancements, NMR data interpretation still faces fundamental physical and environmental obstacles:

- **Surface Relaxivity Heterogeneity:** The surface relaxivity ($\rho$) is often treated as a constant, but it varies significantly across different samples and lithologies. Correcting for this variability is essential for accurate permeability and pore-size modeling.
- **Internal Magnetic Field Gradients:** Paramagnetic minerals, such as iron and manganese, create internal gradients that cause additional dephasing. This effect shifts $T_2$ distributions and can lead to overestimations of the surface-to-volume ratio if not properly accounted for.
- **Rapid Relaxation Limits:** In unconventional shales and heavy oils, signals often relax faster than the tool's minimum echo spacing ($T_e$) or "dead time," leading to missing porosity or "porosity deficit".
- **Operational Dynamics:** In LWD environments, lateral and longitudinal tool motion can degrade vertical resolution and cause signal artifacts that require complex real-time correction algorithms.

### 10.2 Future Directions: Multi-frequency NMR and Digital Integration

The industry is moving toward high-resolution, adaptive systems that integrate NMR more deeply with automated workflows:

- **Ultra-short Echo-Time Sensors:** New tool designs, such as those utilizing **Q-switch techniques**, aim to reduce echo spacing to **0.3 ms or lower**. This allows for the capture of extremely fast-relaxing components in bitumen and nanoporous systems.
- **High-Resolution Multi-frequency Logging:** Next-generation tools are designed for faster logging speeds (up to twice the speed of traditional MRIL tools) while maintaining high signal-to-noise ratios (SNR) through optimized antenna apertures and multi-frequency packing.
- **Full Integration with the "Digital Oilfield":** The future lies in the marriage of NMR with **Artificial Intelligence and Machine Learning**. These technologies will enable adaptive denoising, automated fluid identification on 2D maps, and the development of "Digital Twins" that simulate reservoir behavior under various production scenarios.
- **NMR in the Energy Transition:** Beyond hydrocarbons, NMR is being repurposed for **geostorage investigations of CO$_2$ and H$_2$**, as well as geothermal reservoir evaluation, focusing on the long-term stability and transport mechanisms of low-carbon energy carriers.

### 10.3 Concluding Remarks

Nuclear magnetic resonance has fundamentally changed formation evaluation by providing a direct, **lithology-independent "window"** into the pore space. From characterizing the most complex organic-rich shales to measuring real-time flow in the wellbore without radioactive sources, NMR offers a versatile suite of solutions for the modern upstream industry. As the industry pivots toward smarter, more sustainable management of subsurface resources, the continued development of high-resolution NMR and its integration with AI-driven workflows will be strategic in **maximizing hydrocarbon recovery while minimizing environmental footprints**.

## References

[1] Yuan Y, Rezaee R, Zhou M-F, Iglauer S (2023). A comprehensive review on shale studies with emphasis on nuclear magnetic resonance (NMR) technique. *Gas Science and Engineering* 120:205163. <https://doi.org/10.1016/j.jgsce.2023.205163>

[2] Elsayed M, Isah A, Hiba M, Hassan A, Al-Garadi K, Mahmoud M, El-Husseiny A, Radwan AE (2022). A review on the applications of nuclear magnetic resonance (NMR) in the oil and gas industry: laboratory and field-scale measurements. *Journal of Petroleum Exploration and Production Technology* 12:2747–2784. <https://doi.org/10.1007/s13202-022-01476-3>

[3] Xiao L (2023). *Practical NMR for Oil and Gas Exploration*. Royal Society of Chemistry. <https://doi.org/10.1039/9781782622253>

[4] Coates GR, Xiao L, Prammer MG (1999). *NMR Logging Principles and Applications*. Halliburton Energy Services, Houston, TX.

[5] Tian J, Yue J, Liu X, Sheng J, Wang H (2025). Nuclear Magnetic Resonance (NMR) Quantifies Stress-Dependent Permeability in Shale: Heterogeneous Compressibility of Seepage and Adsorption Pores. *Processes* 13(6):1858. <https://doi.org/10.3390/pr13061858>

[6] Li C, Shen B, Lu L, Pan A, Li Z, Zhu Q, Sun Z (2023). Quantitative Characterization of Shale Pores and Microfractures Based on NMR T₂ Analysis: A Case Study of the Lower Silurian Longmaxi Formation in Southeast Sichuan Basin, China. *Processes* 11(10):2823. <https://doi.org/10.3390/pr11102823>

[7] Tromp RR, Cerioni LMC (2021). Multiphase Flow Regime Characterization and Liquid Flow Measurement Using Low-Field Magnetic Resonance Imaging. *Molecules* 26(11):3349. <https://doi.org/10.3390/molecules26113349>

[8] Chen S, Li L (2018). Wireline, LWD, and Surface NMR Instruments and Applications for Petroleum Reservoir Formation Evaluation. In: Petrova VM (ed) *Advances in Engineering Research*, chapter 1. Nova Science Publishers, Inc.

[9] Bryan J, Kantzas A, Mirotchnik K (2003). Viscosity Determination of Heavy Oil and Bitumen Using NMR Relaxometry. *Journal of Canadian Petroleum Technology* 42(7). <https://doi.org/10.2118/03-07-02>

[10] Zhao T, Ji Y (2022). Gas Diffusion and Flow in Shale Nanopores with Bound Water Films. *Atmosphere* 13(6):940. <https://doi.org/10.3390/atmos13060940>

[11] Bhorkade S (2025). Emerging Applications of AI and Machine Learning in Nuclear Science and Engineering. *Preprints.org*. <https://doi.org/10.20944/preprints202509.2504.v1>

[12] Silletta EV, Vila GS, Domené EA, Velasco MI, Bedini PC, Garro-Linck Y, Masiero D, Monti GA, Acosta RH (2022). Organic matter detection in shale reservoirs using a novel pulse sequence for T₁–T₂ relaxation maps at 2 MHz. *Fuel* 312:122863. <https://doi.org/10.1016/j.fuel.2021.122863>

[13] Webber JBW, Strange JH, et al. (2008). Review of NMR Cryoporometry. *Physics Reports*. <https://doi.org/10.1016/j.physrep.2008.02.001>

[14] Al-Garadi K, El-Husseiny A, Elsayed M, Connolly P, Mahmoud M, Johns M, Adebayo A (2021). A rock core wettability index using NMR T₂ measurements. *Journal of Petroleum Science and Engineering* 208:109386. <https://doi.org/10.1016/j.petrol.2021.109386>

[15] Albusairi M, Torres-Verdín C (2021). Rapid modeling of borehole measurements of nuclear magnetic resonance via spatial sensitivity functions. *Geophysics*. <https://doi.org/10.1190/geo2020-0388.1>

[16] Anand V, Ali MR, Abubakar A, Iglesias JG (2017). Unlocking the potential of unconventional reservoirs through new generation NMR T₁/T₂ logging measurements integrated with advanced wireline logs. *Petrophysics* 58(2):126–140.

[17] Balliet R, Chen S, Callirgos M, Beard D, Li L (2018). New Magnetic Resonance Wireline Sensor for High-Resolution, Faster Logging, and Better Fluid Typing. *SPE Annual Technical Conference and Exhibition*. <https://doi.org/10.2118/191506-MS>

[18] Deng F, Xiong C, Chen S, Xiao L (2020). A method and device for online magnetic resonance multiphase flow detection. *Petroleum Exploration and Development* 47(4):861–870. <https://doi.org/10.1016/S1876-3804(20)60101-7>

[19] Hirasaki GJ, Lo S-W, Zhang Y (2003). NMR Properties of Petroleum Reservoir Fluids. *Magnetic Resonance Imaging* 21(3-4):269–277. <https://doi.org/10.1016/S0730-725X(03)00135-8>

[20] Liao G, Luo S, Xiao L (2021). Borehole Nuclear Magnetic Resonance Study at the China University of Petroleum. *Journal of Magnetic Resonance* 324:106915. <https://doi.org/10.1016/j.jmr.2021.106915>

[21] Luo S, Xiao L, Can L, Jin Y (2022). A Machine Learning Framework for Low-field NMR Data Processing. *Petroleum Science* 19(1):160–171. <https://doi.org/10.1016/j.petsci.2021.10.024>

[22] Luo G, Luo S, Xiao L, Fu S (2023). Progress and Prospect for Machine Learning Applied in NMR Logging. *Cejing Jishu/Well Logging Technology* 47(6). <https://doi.org/10.16489/j.issn.1004-1338.2023.06.001>

[23] Mamoudou S (2025). Application of Nuclear Magnetic Resonance to Investigate Enhanced Oil Recovery and Geostorage of CO₂ and H₂. Dissertation, University of Oklahoma.

[24] Song Y-Q, Kausik R (2019). NMR application in unconventional shale reservoirs – A new porous media research frontier. *Progress in Nuclear Magnetic Resonance Spectroscopy* 112-113:17–33. <https://doi.org/10.1016/j.pnmrs.2019.03.002>

[25] Sun Z, Li Z, Shen B, Zhu Q (2022). NMR technology in reservoir evaluation for shale oil and gas. *Shiyou Shiyan Dizhi/Petroleum Geology and Experiment* 44(5):901–911. <https://doi.org/10.11781/sysydz202205930>
