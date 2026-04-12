# Multiscale NMR Characterization in Shale: From Relaxation Mechanisms to Porous Media Research Frontiers

Long Zhou, Guangzhi Liao, Ruiqi Fan, Rui Mao, Xueli Hou, Nan Li, Zhilong He, Yushu Zhang, Lizhi Xiao

* State Key Laboratory of Petroleum Resources and Engineering, China University of Petroleum (Beijing), Beijing, China
* Research Institute of Exploration and Development, Xinjiang Oilfield Company, PetroChina, Karamay, Xinjiang, China
* China National Petroleum Corporation, Xi'an, Shaanxi, China

**Correspondence:** Guangzhi Liao (liaoguangzhi@cup.edu.cn)

**Received:** 15 March 2025 | **Revised:** 21 July 2025 | **Accepted:** 4 August 2025

**Funding:** This study was supported by the National Natural Science Foundation of China (42474165 and U24B6001-3-1) and the National Key Research and Development Program (2023YFF0714102 and 2019YFA0708301).

**Keywords:** fluids | MRI | NMR | porous media | shale

## ABSTRACT

Nuclear magnetic resonance (NMR), as an advanced non-destructive and rapid measurement technology, has shown unique advantages in the evaluation of unconventional shale reservoirs. This paper systematically introduces the fundamental principles of NMR spectroscopy and relaxation theory in porous media and comprehensively analyzes the experimental techniques and analytical methodologies for NMR and magnetic resonance imaging (MRI) applications in shale characterization. The content aims to provide substantive reference value for researchers and practitioners engaged in shale NMR investigations. Notably, despite existing technical challenges in expository relaxation mechanisms and capturing nanoscale pore signals within shale matrices, NMR applications present novel possibilities for developing emerging research frontiers. For instance, there is significant synergy between NMR technology and emerging fields such as supercritical CO₂ enhanced recovery, carbon capture, utilization and storage (CCUS), and underground hydrogen storage. This paper methodological framework exhibits extensibility to studies of tight sandstones, hydrate-bearing sediments, and other porous media systems, thereby offering cross-disciplinary technical support for establishing carbon neutrality-oriented energy development paradigms.

## 1. Introduction

With global economic expansion, the energy demand has intensified supply-demand imbalances. Shale has been recognized as a significant unconventional resource due to its vast global reserves, drawing widespread research interest worldwide [1]. However, unlike conventional reservoirs (typically characterized by high porosity and good permeability), shale reservoirs exhibit greater complexity. They are rich in organic matter (e.g., bitumen and kerogen), possess intricate inorganic mineral compositions (e.g., clay, quartz, feldspar, calcite, and pyrite), and feature diverse pore types (predominantly nanopores). These characteristics lead to low permeability, low porosity, poor pore connectivity, and marked heterogeneity [2-4]. Nanopores in shale serve as both the primary storage spaces for oil and gas and critical flow pathways. Within these nanopores, fluid-rock and fluid-fluid interactions lead to elevated fluid viscosity and density near the pore surface. Consequently, fluids in shale exhibit complex multiphase flow behavior that significantly differs from that observed in conventional reservoirs [5]. The interplay of intricate fluid-rock interactions, nanoscale pore structures, and mineral compositions governs the physical properties of the fluids, as well as the distribution and mobility of oil and water [6]. These factors pose significant challenges to shale oil reservoir research.

The successful commercialization of shale oil and gas production fundamentally depends on the accurate evaluation of reserves and fluid transport capabilities within complex nanopore structures, along with the dynamic monitoring of shale reservoirs [7, 8]. Current experimental techniques for studying fluids and pore characteristics in shale encompass both qualitative analyses and measurements, including stepwise pyrolysis, scanning electron microscopy (SEM), mercury intrusion capillary pressure (MICP), small-angle and ultra-small-angle neutron scattering (SANS/USANS), low-temperature nitrogen adsorption (LTNA), theoretical calculations, and numerical simulations [9-16]. Nevertheless, these methods face limitations in detection scale, making it challenging to achieve high-resolution characterization of rapidly relaxing fluids within shale nanopores. Moreover, significant discrepancies exist among the results obtained using different methods, and a standardized technique for accurately quantifying the structural properties of shale nanopores has yet to be established [17].

Among the experimental techniques employed in shale research, nuclear magnetic resonance (NMR) technology has attracted considerable attention for its ability to rapidly and accurately characterize the internal information of shale. Unlike optical measurement methods, NMR employs a unique mechanism that effectively characterizes both the fluid composition and the pore structure of core samples, while being less constrained by experimental conditions [18, 19]. For petroleum engineers and researchers, NMR studies in shale have primarily focused on three aspects:

1. Gaining a comprehensive understanding of NMR principles;
2. Interpreting NMR signals based on relaxation theories and porous media models; and
3. Analyzing specific relaxation mechanisms in shales through the integration of petrological characteristics and fluid properties.

Given the rapid development and widespread application of NMR technology in unconventional shale. The paper begins with a systematic introduction to the fundamental principles of NMR spectroscopy and NMR theory in porous media. Subsequently, it presents an integrated analysis of NMR applications and methodologies in multiscale, high-resolution petrophysical characterization of shale, systematically elucidating the mechanisms of fluid–pore interactions under nanoscale confinement effects. Particularly, this paper analyzes the current limitations of NMR technology in shale research and proposes potential future research directions to overcome these challenges, thereby providing valuable theoretical support for the production potential of shale reservoirs.

## 2. Basic Principles of NMR Spectroscopy

NMR is a physical phenomenon occurring in systems with atomic nuclei possessing magnetic moments and angular momentum [20]. It correlates the decay of measured signals with the interactions between nuclear spins and their surroundings, as well as interactions among the nuclear spins themselves [21]. Also known as Laplace inversion NMR, it is widely applied to monitor molecular dynamics, even within complex materials [22, 23].

The fundamental principle of NMR spectroscopy involves the detection of signals as protons (e.g., hydrogen nuclei) undergo excitation by RF waves within a magnetic field environment (Figure 1A) [21]. Within the spectrometer apparatus, a magnet unit produces a permanent magnetic field $B_0$. The hydrogen nuclei spins in the sample (with and against the field) undergo a shift in their average orientation as a result of field-induced polarization, generating a net magnetization vector $M_0$ aligned with $B_0$. When RF pulses are introduced, they create a magnetic field $B_1$ positioned orthogonally to $B_0$. These applied pulses subsequently reorient the net magnetization from its initial state $M_0$ to $M$ [24].

![Figure 1](placeholder_for_figure1.png)
**Figure 1:** (A) Schematic diagram of an NMR instrument. (B) Illustration of the basic operating principle of NMR. (C) Correlation of T1 and T2 relaxation times with molecular mobility, temperature, and sample states.

In solids or liquids, nuclear spins tend to exhibit an excess alignment in the same direction as the static magnetic field $B_0$. While bar magnets align completely parallel or antiparallel to a magnetic field, atomic nuclei with intrinsic spin have their spin angular momentum components along the $B_0$ direction quantized into discrete energy levels, and they precess around the $B_0$ axis (Figure 1B). This behavior is often likened to the wobbling motion of a gyroscope under the influence of Earth's gravitational field, providing an analogy for the concept of spin, which is fundamentally a quantum mechanical phenomenon. The rate of this precession around the magnetic field direction is referred to as the Larmor frequency, which is proportional to the strength of the magnetic field and is described by the Larmor Equation (1) [25]:

$$ \omega = \gamma B_0 \tag{1} $$

where $\omega$ denotes the protons angular frequency; $\gamma$ represents the gyromagnetic ratio, which is a constant specific to a given type of nucleus; and $B_0$ indicates the strength of the magnetic field. Upon cessation of radiofrequency (RF) excitation, the excited spin system undergoes energy dissipation through relaxation phenomena. The dissipation of this excited-state magnetization constitutes spin relaxation, which occurs through two distinct mechanisms: spin-lattice relaxation ($T_1$) and spin-spin relaxation ($T_2$) (Figure 1B). These complementary processes are quantitatively described by first-order rate equations governed by characteristic time constants: $T_1$ for energy transfer to the lattice and $T_2$ for coherent signal loss (Figure 1C) [26].

As previously established, the sample's spins are excited by RF pulses before measuring signal attenuation. Studies of relaxation time distributions range from one-dimensional (1D) measurements (e.g., $T_1 / T_1^*$ and $T_2 / T_2^*$) to more complex multi-dimensional experiments (e.g., $T_1 - T_2$, $D - T_2$, and $T_1 / T_1^* - T_2^*$), employing various pulse sequences to acquire time-domain signals (e.g., IR, CPMG, PFG, IR/SR-CPMG, and IR/SR + PFG-CPMG) [27-30]. Notably, in two-dimensional (2D) or multi-dimensional NMR, sequence designs aim to combine 1D pulse sequences with specific echo times and gradient applications [21]. $T_1$ and $T_2$ relaxations are both first-order processes, and therefore, they can be described by exponential decay functions with characteristic relaxation time distributions. The acquired data are evaluated or deconvoluted using two methods: discrete multiexponential fitting or inverse Laplace transform (ILT) methods [31]. The ILT method presents relaxation times in the form of continuous distributions.

## 3. Brief Description of Relaxation Theory for Porous Media

The relaxation characteristics of confined liquid in porous materials depend on pore size as well as interactions between the pore wall and liquid. The Bloembergen-Purcell-Pound (BPP) theory relates relaxation time with the correlation time of dipolar interactions [32], as expressed by Equations (2) and (3) [33]:

$$ \frac{1}{T_1} = 2C\left(\frac{2\tau}{1 + \omega^2\tau^2} +\frac{8\tau}{1 + 4\omega^2\tau^2}\right) \tag{2} $$

$$ \frac{1}{T_2} = C\left(3\tau +\frac{5\tau}{1 + \omega^2\tau^2} +\frac{2\tau}{1 + 4\omega^2\tau^2}\right) \tag{3} $$

where $T_1$ represents the time constant for the recovery of the longitudinal component ($M_z$) of the nuclear spin magnetization vector $M_0$ along the external magnetic field $B_0$ (typically designated as the $z$-axis) to its thermal equilibrium value; $T_2$ is the time constant describing the decay of the transverse component ($M_{xy}$) of the nuclear spin magnetization vector $M$ perpendicular to the external magnetic field $B_0$ (Figure 1B); $\omega$ is the Larmor frequency; $\tau$ is correlation time; and $C$ is a constant.

According to surface relaxation theory, the $^1$H relaxation time measurements primarily arise from the contributions of both bulk relaxation and surface relaxation of the liquid confined within the pores. Assuming rapid molecular exchange or diffusion between the liquid inside the pore and the liquid layer adsorbed on the pore surface, the exchange time between liquid on the surface and liquid in the pore will be short [34]. The relaxation time rates will be exhibited by all the confined liquid as expressed by a weighted average of the bulk ($T_B$) and surface relaxation ($T_S$) rates, as shown by Equations (4) and (5) [35-37]:

$$ \frac{1}{T_1} = \frac{f_s}{T_{1S}} +\frac{f_b}{T_{1B}}\approx \frac{1}{T_{1B}} +\frac{S_p}{V_p}\frac{\epsilon_S}{T_{1S}} = \frac{1}{T_{1B}} +\rho_1\frac{S_p}{V_p} \tag{4} $$

$$ \frac{1}{T_2} = \frac{f_s}{T_{2S}} +\frac{f_b}{T_{2B}} +\frac{D\gamma^2G^2T_E^2}{12}\approx \frac{1}{T_{2B}} +\frac{S_p}{V_p}\frac{\epsilon_S}{T_{2S}} +\frac{D\gamma^2G^2T_E^2}{12} = \frac{1}{T_{2B}} +\rho_2\frac{S_p}{V_p} +\frac{D\gamma^2G^2T_E^2}{12} \tag{5} $$

where $f_b$ and $f_s$ represent the volume fractions of the bulk and surface layers of the pore, respectively; with $f_b + f_s = 1$; $\rho_1(\equiv \epsilon_s / T_{1S})$ and $\rho_2(\equiv \epsilon_s / T_{2S})$ are the surface relaxivity for surface relaxation times of $T_{1S}$ and $T_{2S}$ which express the strength of surface relaxation; $\epsilon_S$ is the thickness of adsorbed liquid layer on the pore surface; $S_p$ and $V_p$ are the pore surface area and volume, respectively; $D$ and $G$ are the diffusion coefficient and magnetic field gradient strength, respectively; and $T_E$ is the experimental inter-echo time for $180^\circ$ pulses [33].

![Figure 2](placeholder_for_figure2.png)
**Figure 2:** Schematic diagram showing the relaxation mechanisms of hydrogen-containing components in shale [38, 39].

Various proton relaxation behaviors in shale have their corresponding relaxation mechanisms (Figure 2). Surface relaxation arises from transient local field inhomogeneity caused by changes in electron density when gas spins and surface spins undergo dipolar coupling near charged surfaces, as well as from interactions with other spins (such as $^{13}$C and $^{1}$H) on mineral surfaces and interactions with paramagnetic ions on surfaces. Bulk relaxation is caused by phase shifts induced by intermolecular interactions (liquids) and intramolecular spin interactions (gases). Diffusion relaxation is induced by inhomogeneous magnetic fields, leading to variations in the Larmor frequency range and resulting in phase loss [38, 39]. Note that $^{1}$H diffusion in shale belongs to the free diffusion regime with high probability, but diffusion-enhanced relaxation is unlikely to occur under the relevant experimental parameters of the CPMG pulse sequence in the presence of weak internal gradients [40]. According to the observed relaxation time magnitude and the known pore size distribution in shale, it is widely accepted that the dominant relaxation mechanism for fluids in shale is surface relaxation [34]. Surface relaxation is closely related to the lithological properties of shale. Variations in shale lithology, such as differences in mineral composition (e.g., clay content), paramagnetic mineral content (e.g., pyrite), organic matter type (kerogen, bitumen, etc.), and their distribution within the pore network, may result in different NMR responses for the same fluid [41, 42].

However, due to the complex composition of shale, there is still controversy regarding the dominant relaxation mechanism, specifically whether the measured relaxation time of hydrogen-containing fluids is primarily influenced by surface relaxation within pores or by other factors [22, 43, 44]. Further research is needed to distinguish and clarify the contributions of different relaxation mechanisms in various hydrogen-containing components within shale.

## 4. Multiscale NMR Characterization in Shale

NMR, as a non-destructive and rapid technique, offers distinct advantages in probing the microscopic properties of rocks. It provides insights into pore structure, fluid properties, and their mutual interactions. In shale research, NMR technology has been employed for various aspects, including fluid typing [45-48], pore size distribution (PSD) characterization [49-52], and wettability evaluation. Table 1 summarizes the relaxation times and main research directions of different hydrogen-containing components in shale.

### 4.1 Fluid Typing

Fluid typing is a critical step in shale oil and gas evaluation. For complex lithological shale, 2D NMR technology has become a key tool for fluid identification due to its ability to overcome the limitations of 1D NMR in separating overlapping signals (Figure 3) [53]. Currently, $T_1 - T_2$ maps are widely used for fluid typing in shale. Figure 3 shows a typical 2D NMR $T_1 - T_2$ map for fluid components in shale reservoirs. Generally, regions with lower $T_1/T_2$ ratios correspond to bulk fluids (free water, light oil, gas) and clay-bound water, while regions with higher $T_1/T_2$ ratios are associated with viscous fluids (heavy oil, bitumen) and organic matter (kerogen) [22, 27, 53].

![Figure 3](placeholder_for_figure3.png)
**Figure 3:** Typical 2D NMR T1-T2 mapping for fluid components in shale reservoirs [22, 27, 53].

![Figure 4](placeholder_for_figure4.png)
**Figure 4:** (A) T1-T2 NMR experimental process of shale after water and oil restoration, and oil-saturated [54]. (B) NMR T2 projection spectra decomposition of shale samples from the Gulong Sag in the Songliao Basin of China [55]. (C) T1-T2 mapping of different hydrogen-containing components in pores of high-maturity lacustrine shale [56].

Zhang et al. [54] employed NMR technology to study the microscopic occurrence and distribution characteristics of oil and water in in-situ shale. By analyzing the state of fluids in shale cores after restoration to original formation conditions and after oil saturation, they distinguished the distribution areas of movable oil and irreducible oil, movable water and irreducible water, as well as organic matter on the 2D map (Figure 4A). Wang et al. [55] proposed a method for identifying fluid properties by decomposing the NMR $T_2$ spectrum of shale. By fitting the $T_2$ spectrum with Gaussian functions, they decomposed it into multiple components corresponding to different fluids (clay-bound water, capillary-bound water, movable water, viscous oil, and movable oil), enabling quantitative evaluation of oil and water content and their mobility (Figure 4B). Chi et al. [56] used 2D NMR technology to study the occurrence characteristics of oil in high-maturity lacustrine shale pores. They further subdivided the oil components into adsorbed oil, trapped oil, and free oil based on their $T_1$ and $T_2$ relaxation characteristics (Figure 4C).

It is important to note that due to significant differences in the properties of organic matter and fluids in shales from different regions, as well as variations in instrument parameters (e.g., magnetic field strength, echo time), there is currently no unified standard for 2D NMR fluid typing in shale. The presence of kerogen and bitumen, with their high $T_1/T_2$ ratios, can potentially obscure the signals of heavy oil. Conversely, when clay-bound water exhibits very short relaxation times, it may also be misinterpreted as solid organic matter. Additionally, the high viscosity of bitumen can lead to overlapping signals with kerogen, making it difficult to separate relaxation responses between fluids and solids [22]. The use of universal mapping may lead to errors in fluid type determination.

To address the problem that fast spin-spin ($T_2$) signal decay from these solid/viscous components cannot be distinguished and captured, it is recommended to employ specifically designed pulse sequences to enhance signal identification capability and data accuracy. For instance, Panattoni et al. [57] developed a shale characterization method based on CPMG, solid echo (SE), and magic echo (ME) pulse sequences to refocus $^{1}$H-$^{1}$H dipolar coupling, providing detailed differentiation of signal contributions from immobile organic matter and clay-bound water. Future research may focus on broadening the understanding of the multifarious NMR responses of the materials resident in the shale and developing quantitative analysis methods suitable for complex porous media conditions.

### 4.2 Dynamic Oil-Water Migration Mechanisms in Shale

The migration process of oil and water molecules in shale reservoir pores is extremely complex. Factors such as temperature variations, capillary pressure gradients, and physicochemical interactions between fluids and pore walls control the spatial distribution patterns of oil and water in nano- to micro-scale pore systems [58], which directly affect fluid mobility and recovery efficiency. Therefore, gaining a deep understanding of the flow mechanisms of oil and water molecules in shale pores is of great significance.

NMR technology has been applied to the study of fluid flow mechanisms and mobility evaluation in shale due to its non-destructive detection advantages for fluid dynamic behavior. Liu et al. [59] employed NMR relaxometry to study the changes of water and oil behaviors in Chinese lacustrine Qingshankou shales under different saturated states (imbibition and evaporation without pressure). The differences between regions on $T_1 - T_2$ maps show dynamic migrations of water and oil in pores during the imbibition and evaporation (Figure 5A) [59]. Lin et al. [60] utilized NMR to further investigate the flow patterns of shale oil and the distribution of remaining oil during spontaneous imbibition in three types of shales. They proposed the flow mechanism during water flooding, where MnCl₂ solution enters hydrophilic micropores and displaces oil from micropores into hydrophobic mesopores (Figure 5B). This part of the oil can be transmitted to the microfractures through the mesopores by the capillary pressure difference on both sides of the micropores and discharged [60].

![Figure 5](placeholder_for_figure5.png)
**Figure 5:** (A) T1-T2 maps during imbibition and evaporation processes [59]. (B) Schematic diagram of oil-water flow mechanism during spontaneous imbibition [60]. (C) T1-T2 maps of laminated shale samples from different depths [61].

Shale oil movability evaluation is the micro-scale static characterization of the occurrence state and flow threshold of oil in the reservoir space, namely, under the original formation conditions [62]. It emphasizes the depiction of the in situ state and the quantitative compositional determination of oil within the micro-nano pore-fracture system, which is primarily controlled by fluid-solid interaction forces in the confined micro-nano domain. Sun et al. [61] conducted a comprehensive assessment of shale oil mobility in lacustrine formations through NMR, Rock-eval pyrolysis, and multi-temperature pyrolysis methods (Figure 5C). Laminas consist of bright laminae that contain high contents of quartz, feldspar, calcite, and dolomite, which are frequently intercalated with dark laminae dominated by clay and organic matter [61]. They suggested that the movable oil content in bright layers is generally higher than that in the dark layers, attributed to the migration of oil generated in the dark layers to bright layers. However, with thermal maturity evolution, the physical properties and fluid occurrence states of oil may change, thereby affecting its mobility. This dynamic evolution process should become an important direction for future research.

Although the aforementioned work has demonstrated the reliability of NMR technology in revealing fluid flow behavior in shale, further research is still needed to enhance its applicability in shale reservoir mobility analysis and interpretation. This may include detailed studies on individual mineral wettability to gain a deeper understanding of the microscopic mechanisms of oil-water migration behavior under different wettability conditions. Additionally, expanding methods for imbibition and displacement testing under high-pressure and high-temperature conditions to reveal non-steady-state fluid flow mechanisms under extreme environments and complex stress conditions.

### 4.3 Evaluation of Fluid Saturation and Viscosity Under Shale Matrix-Fluid Interface Coupling

Estimation of fluid saturation is critical for both reserve calculation and perforation location selection in shale reservoirs. According to surface relaxation theory (Equation 5), the $T_2$ spectrum reflects the pore radius distribution in shale reservoirs. Under the combined influences of capillary forces and viscosity, when the pore radius decreases to a threshold, the fluid becomes trapped and ceases to flow. This critical pore radius corresponds to a specific value in the $T_2$ spectrum, commonly referred to as the $T_2$ cutoff, which serves as a key parameter for calculating bound fluid saturation and assessing fluid mobility [63]. Since capillary force-based studies provide an effective method for evaluating the initial fluid saturation of oil and gas reservoirs, it is also possible to convert NMR $T_2$ spectra into capillary pressure curves and then use capillary pressure-based methods to calculate shale fluid saturation [64]. The accuracy of this method depends on the precise calculation of pore structure using NMR data. Specifically, the $T_2$ geometric mean and interval porosity corresponding to spectral peaks exhibit positive and negative correlations with oil saturation, respectively. Zhang et al. proposed a new shale oil saturation calculation model (Equation 6) based on the analysis of the relationship between NMR $T_2$ geometric mean and interval porosity with oil saturation [65]. This model yields a root mean square error (RMSE) of $5.85\%$ between calculated and measured oil saturation results, reflecting the accuracy and effectiveness of this method.

$$ S_o = k\frac{T_{2\text{gm}}^a}{A_{p1}^b} \times 100\% \tag{6} $$

where $T_{2\text{gm}}$ is $T_2$ geometric mean; $A_{p1}$ is interval porosity; and $k$, $a$, and $b$ are empirical constants.

Other 2D NMR methods, such as $T_1 - T_2$ maps, can also be used to quantitatively evaluate fluid saturation. By comparing the 2D maps of shale samples under different saturation states (e.g., as-received, extracted, and re-saturated), the volumes of different fluid components (e.g., clay-bound water, movable water, oil, bitumen) can be quantified [66-69]. However, quantifying the light hydrocarbon content in shale remains challenging due to its high volatility and potential loss during core retrieval and sample preparation [70, 71]. Future work should focus on developing more accurate methods for preserving and measuring light hydrocarbons in shale.

Viscosity is another critical fluid property that directly impacts producibility. NMR relaxation times are sensitive to fluid viscosity, and empirical relationships have been established to estimate oil viscosity from NMR measurements in conventional reservoirs. However, the application of these relationships to shale is complicated by the presence of organic matter and the dominance of surface relaxation. In shale, the measured relaxation time of oil is a combination of its intrinsic bulk relaxation (related to viscosity) and surface relaxation effects (related to pore size and surface interactions). Decoupling these effects is essential for accurate in-situ viscosity estimation. Several studies have attempted to address this challenge. For example, Yang et al. [48] proposed a method for determining the in situ viscosity of fluids in porous media using physical simulation and NMR. They found that the viscosity distribution of water is heterogeneous, rather than constant, likely due to variations in the distance between the rock walls and the water. Moreover, under the same centrifugal pressure, the average in situ viscosity of water increases as more water is separated from the rock, which is likely linked to an increase in permeability.

NMR evaluation of shale oil viscosity can be calculated by Equation (7) [72]:

$$ \eta_o = \frac{aT}{T_{2(\text{LM})} \times f(\text{GOR})} \tag{7} $$

where $\eta_o$ is the oil viscosity, $T$ is the oil temperature in Kelvin, $T_{2(\text{LM})}$ is the $T_2$ geometric mean; $f(\text{GOR})$ is a dimensionless function of the gas-oil ratio, and $a$ is equal to $0.004 \, \text{s} \, \text{cp} \, \text{K}^{-1}$.

To clarify the effectiveness of NMR relaxometry in assessing shale fluid viscosity, researchers have discussed the correlation between viscosity and relaxation. Sandor et al. utilized temperature and echo time (TE) dependent $T_1$ and $T_2$ relaxation time and hydrogen index (HI) data to predict heavy oil viscosity [73]. They demonstrated that NMR relaxometry measurements exhibit excellent correlation with heavy oil viscosity. Since the primary relaxation mechanism in shale reservoirs is surface relaxation, the decay rate is primarily a function of surface-fluid interactions and does not represent the internal reactions of fluid protons. However, Cao et al. proposed, based on simulation studies, that the relaxation of heavy oil (with a viscosity greater than $100 \, \text{cps}$) would be controlled by internal reactions [74]. Therefore, following the procedure proposed by Bryan et al. for estimating oil sand viscosity, heavy oil viscosity can be related to its relaxation time [75]. Tinni et al. also utilized the $T_1/T_2$ ratio to distinguish between movable and immovable fluids in both conventional and unconventional reservoirs [76]. Experimental results showed that non-flowing hydrocarbons are unaffected by surface relaxation, a finding confirmed by Cao et al. in their simulations [74]. This suggests that NMR can at least qualitatively estimate fluid viscosity.

### 4.4 NMR Characterization of Nanoscale Pore Structures and Low Permeability

Porosity and permeability are fundamental petrophysical properties that control fluid storage and flow in shale reservoirs. Porosity ($\phi$) quantifies the total pore space within the rock matrix. Permeability ($K$) measures the ease with which fluids can flow through the pore network under a pressure gradient. Shale is a typical low-permeability reservoir with a complex pore structure; its permeability generally ranges from $10^{-2}$ to $10^{-7}$ millidarcies (mD) and is closely related to pore geometry (Figure 6) [77, 78]. The micropores and nanochannels in shale are highly developed, with complex pore size distributions and porosity generally less than $10\%$ [79].

![Figure 6](placeholder_for_figure6.png)
**Figure 6:** (A) Proposed pore network structure for simulation of gas transport within organic-rich shale deposits, including two types of micro porosities (clay and kerogen), meso-pores, and fractures entangled with solid minerals such as pyrite, calcite, and silicate [77]. (B) Schematic diagram illustration for the changes of pore networks of shale after reacting with different fracturing fluids: (a) pretreatment shale; (b) slick water-treated shale; and (c) cross-linked gel-treated shale [78].

Among the various methods used to study the pore structure of shale, NMR has the advantage of detecting micro-nanopores in complex porous media. Based on the $^1$H relaxation mechanism, the total relaxation signal is proportional to the total number of $^1$H nuclei, and the integral area of the $T_2$ spectrum is proportional to the hydrogen content in the pores of the core sample, which is equal to the pore volume within the samples. The porosity of the shale can be determined by parametrically calibrating the $T_2$ relaxation signal to the pore volume (Equations 8 and 9) [80]:

$$ y = kx + b \tag{8} $$

$$ \phi = \frac{V'}{V} \times 100\% = \frac{y}{V} \times 100\% \tag{9} $$

where $x$ is the integral area of the $T_2$ spectrum; $y$ is the water volume of the standard sample; $k$ is the slope of the standard equation; $b$ is the vertical intercept of the standard equation; $\phi$ is the NMR porosity, $\%$; $V'$ is the pore volume of the sample, $\text{cm}^3$; and $V$ is the total volume of the sample, $\text{cm}^3$.

To evaluate the accuracy of NMR-based porosity measurements in shale, this study compiled porosity data from various measurement methods across different shale blocks (Table 2) and conducted systematic comparative analysis. As shown in Table 2, significant differences exist between porosity values obtained from NMR relaxometry and those from MICP and LTNA methods. These discrepancies may stem from the limitations of each method in their optimal pore size detection ranges. NMR relaxation signals consider all spins from fluids contained in all types of pore spaces, while the LTNA method effectively measures pore sizes ranging from 2 to $50 \, \text{nm}$. MICP methods are destructive and only detect pore throat volumes, primarily obtaining meso- and macro-porosity, with their optimal detection range being $100 \, \text{nm}$ to $100 \, \mu\text{m}$. In contrast, NMR-measured porosity shows smaller differences compared to other methods; particularly $\phi_{\text{NMR}}$ and $\phi_{\text{He}}$ demonstrate high consistency, indicating that helium-measured porosity can also detect all pores.

Zhao et al. analyzed the NMR porosity measurement results of shale samples under different water soaking times [90]. The results showed that NMR-based porosity values of shale samples vary with changes in water soaking time. This raises the question: When water molecules are used as probes in NMR methods, can they objectively and accurately reflect the true porosity of shale samples?

Shale porosity measured by NMR is commonly used to calculate permeability and evaluate reservoirs [81]. After identifying the fluid type and knowing the total porosity, the fluid density can be estimated using the HI Equation (10) [37, 91]:

$$ \text{HI} = \frac{\phi_X}{\phi_T} = \frac{9\rho_X N_H}{M_X} \tag{10} $$

where $\phi_X$ represents the measured porosity; $\phi_T$ is the true porosity after HI correction; $\rho_X$ is the fluid density $(\text{g}/\text{cm}^3)$; $N_H$ is the number of hydrogen atoms in the chemical structure; and $M_X$ is the molecular weight.

The two most widely used NMR permeability models are the Schlumberger Doll Research (SDR) model and the Timur-Coates model:

$$ K_{\text{SDR}} = C_{\text{SDR}} \times \phi_{\text{NMR}}^m \times T_{2(\text{LM})}^n \tag{11} $$

$$ K_{\text{Coates}} = \left(\frac{\phi_{\text{NMR}}}{C_{\text{Coates}}}\right)^{m_2} \times \left(\frac{\text{FFI}}{\text{BVI}}\right)^{n_2} \tag{12} $$

$$ T_{2(\text{LM})} = \exp \left(\frac{\sum \ln(T_{2i}) \phi_i}{\sum \phi_i}\right) \tag{13} $$

where $K_{\text{SDR}}$ and $K_{\text{Coates}}$ represent the matrix permeability; $\phi_{\text{NMR}}$ is the total NMR porosity; $T_{2(\text{LM})}$ is the average of the log of the NMR $T_2$ spectrum under water-saturation conditions ($S_w = 100\%$); FFI (free fluid index) and BVI (bound volume irreducible) are the volume of mobile fluid and irreducible bound fluid detected by NMR, respectively; $T_{2i}$ is the $i$-th transverse relaxation time (s); $\phi_i$ is the incremental NMR porosity corresponding to the $i$-th transverse relaxation time; $\sum \phi_i$ is the total NMR porosity; and $m, n, C_{\text{SDR}}, m_2, n_2, C_{\text{Coates}}$ are model parameters that can be obtained experimentally. When the number of cores is large and cannot be determined, the default values of 4, 2, 10, 4, 2, 10 can be used. However, due to the complex aggregate structure of shale, the SDR and Coates models of shale need to be modified in combination with the NMR pore structure characteristics [92]:

$$ K_{\text{SDR}} = a \log (T_{2(\text{LM})}) + b \log \phi_{\text{NMR}} + C_{\text{SDR}} \tag{14} $$

$$ K_{\text{SDR}} = a \log (T_{2(\text{LM})}) + b \log \phi_{\text{eff}} + C_{\text{SDR}} \tag{15} $$

$$ \log K_{\text{Coates}} = a \log \left(\frac{\text{FFI}}{\text{BVI}}\right) + b \log \phi_{\text{NMR}} + C \tag{16} $$

$$ \log K_{\text{Coates}} = a \log \left(\frac{\phi_{\text{eff}}}{\text{BVI}}\right) + b \log \phi_{\text{eff}} + C \tag{17} $$

where $\phi_{\text{NMR}}$ and $\phi_{\text{eff}}$ represent the total NMR porosity and the effective NMR porosity, respectively, while CBW refers to the volume of clay-bound water [93]; $C_{\text{SDR}}$ is a correction coefficient related to formation type, typically determined through the calibration of porosity and permeability in core samples. However, the modified SDR model is only applicable to rocks that are fully saturated with water, whereas the modified Coates model can also be applied to rocks containing both water and hydrocarbons, using the $T_2$ cutoff value [94]. Although several new shale NMR permeability models based on different theories and applicable to different blocks have been established for unconventional reservoirs (Table 3), the accuracy of these models still requires more laboratory analytical testing for validation due to the extreme complexity and heterogeneity of pore geometry in shale.

*Table 2: Shale porosity data from different blocks obtained by NMR and other measurement methods. (Refer to original PDF for the detailed table)*

*Table 3: NMR permeability models for shale reservoirs based on different theories. (Refer to original PDF for the detailed table)*

### 4.5 Conversion of Pore Size Distribution via Surface Relaxivity and NMR Cryoporometry

Assuming pores are cylindrical or spherical, the $T_2$ relaxation time can be mapped to pore diameters using the surface relaxivity (SR) conversion coefficient (Equation 18). This allows the pore size distribution (PSD) to be derived from NMR $T_2$ spectra. SR is a critical parameter that characterizes the relaxation intensity at the solid-fluid interface and is the most important fitting parameter when converting NMR signals into length scales. Figure 2 illustrates the surface relaxation characteristics in shales. To determine SR in shale, several calculation methods based on the NMR principle shown in Equation (18) have been developed (Equations 19-23).

$$ \frac{1}{T_{2i}} = \rho_2 \left(\frac{S}{V}\right)_i = \rho_2 \left(\frac{C_s}{r_i}\right) \tag{18} $$

where $T_{2i}$ is the transverse $T_2$ relaxation time at experimental point $i$; $\rho_2$ is the surface relaxivity; $(S/V)_i$ is the ratio of pore surface area to pore volume ratio (surface-to-volume ratio [SVR]) that can be determined by low-pressure gas adsorption experiment; $r_i$ is the pore radius at experimental point $i$; and $C_s$ is a shape factor—a constant value dependent on geometric pore shape. Slit, cylindrical, and spherical pores correspond to values of 1, 2, and 3, respectively [82].

**Determination of SR with LTNA** [98, 99]:

$$ \rho_2 = \frac{1}{S_p T_{2-n}} \tag{19} $$

$$ T_{2-n} = \exp \left(\frac{\sum_{i=2\min}^{100*T_{2\min}} \phi_i^* \ln(T_{2i})}{\sum_{i=2\min}^{100*T_{2\min}} \phi_i}\right) \tag{20} $$

where $S_p$ is obtained by using the BET method. The maximum $T_2$ time is assigned to $100*T_{2\min}$ in this equation because of the 2-200 nm range of pore diameters provided by LTNA.

**Determination of SR with MICP** [82, 99]:

$$ r = \frac{2\sigma \cos \theta}{P_c} \tag{21} $$

$$ \rho_2 = \frac{r}{T_{2i} C_s} \tag{22} $$

where $P_c$ is the capillary pressure, $\sigma$ is the surface tension of the fluid, $\theta$ is the contact angle, and $r$ is the pore throat radius determined from MICP.

**Determination of SR using Diffusion NMR** [102-104]:

$$ \rho_2 = \frac{D}{\lambda T_{2S}} \tag{23} $$

where $\lambda$ is the surface interaction parameter.

The surface relaxivity of shales is usually lower than that of conventional sandstones and carbonates [100]. The reason may be that a large number of organic pores develop in shale, and the surface relaxivity of organic matter is significantly lower than that of clay minerals. The organic matter in shale has weaker surface relaxation effects on fluids compared to clay minerals [105]. Saidian and Prasad [98] also pointed out that SR varies with the concentration of paramagnetic minerals in the sample. The higher the concentration of paramagnetic minerals, the higher the SR. Therefore, when using Equation (18) to convert $T_2$ spectra into PSD, it is crucial to consider the differences in SR of different shale components and the impact of paramagnetic minerals.

Due to the complex mineral composition of shale and the presence of multiple pore types (organic pores, inorganic pores, microfractures), its PSD exhibits multi-scale characteristics. Reliable conversion of NMR $T_2$ spectra to PSD requires accurate determination of SR. However, direct measurement of SR in shale is challenging. Current methods often combine NMR with other techniques (e.g., MICP, LTNA, SEM) to estimate an average SR for the entire core [106, 107]. This average SR may not accurately represent the relaxation behavior in different pore types, leading to uncertainties in the converted PSD. Furthermore, the assumption of a constant SR across the entire pore size range may be invalid, as SR itself might be pore-size dependent [101].

NMR cryoporometry (NMRC) is another powerful technique for measuring PSD in porous media, particularly for nanopores [108-111]. It is based on the melting point depression of a confined liquid, which is inversely proportional to the pore size. By cooling a liquid-filled porous material and observing the melting behavior using NMR, the PSD can be determined. The relationship between pore diameter ($d$) and melting point depression ($\Delta T_m$) is given by the Gibbs-Thomson equation:

$$ \Delta T_m = T_m - T_m(d) = \frac{4\sigma_{sl} T_m}{d \Delta H_f \rho_s} \tag{24} $$

where $T_m$ is the bulk melting point, $T_m(d)$ is the melting point of the confined liquid, $\sigma_{sl}$ is the surface energy at the solid-liquid interface, $\Delta H_f$ is the bulk enthalpy of fusion, and $\rho_s$ is the density of the solid.

NMRC offers several advantages for shale PSD characterization. It is non-destructive and can probe a wide range of pore sizes, from micropores to macropores, by using different probe liquids with varying melting points and interaction parameters [112]. Common probe liquids include water, cyclohexane, and octamethylcyclotetrasiloxane (OMCTS). OMCTS, with its larger molecular size and weaker interaction with pore surfaces, is particularly suitable for characterizing nanopores in shales as it is less affected by surface interactions and can access pores that might be inaccessible to water [113-116].

Figure 7 shows NMR Cryoporometry PSD using different probe liquids for shale core samples from the Long-Ma-Xi Formation in China [115, 116]. The results demonstrate that NMR Cryoporometry can effectively characterize the full-range PSD of shale, providing valuable insights into the complex pore structure.

![Figure 7](placeholder_for_figure7.png)
**Figure 7:** NMR Cryoporometry PSD using different probe liquids for shale core samples from the Long-Ma-Xi Formation in China [115, 116].

### 4.6 Wettability and Fluid-Rock Interactions

Wettability is a fundamental petrophysical property that governs the distribution, flow, and transport of fluids in porous media. It significantly influences capillary pressure, relative permeability, imbibition pressure, and ultimately the efficiency of oil recovery [117-120]. Particularly in shales with complex nano-scale pore structures, wettability often exhibits significant spatial heterogeneity, which can profoundly impact fluid distribution, flow behavior, and recoverability [121].

In shale reservoirs containing two or more immiscible fluids (e.g., water and oil), wettability describes the interaction strength of fluid molecules with the solid surface, particularly at the pore–solid interface [122, 123]. Wettability reflects the spreading, adsorption, and interfacial tension changes of fluids on solid surfaces. Based on the chemical composition, physical structure, and mineral composition of solid particle surfaces in rocks, wettability can be classified into homogeneous and heterogeneous. Homogeneous wettability refers to spatially uniform chemical and physical properties of the rock surface, resulting in consistent wetting behavior; heterogeneous wettability arises from localized variations in the rock surface (e.g., pore distribution, mineral composition changes), leading to nonuniform wettability [124]. Furthermore, wettability can also be classified into water-, neutral-, and oil-wet based on the degree of wetting [125].

To determine the wettability of shale reservoir, contact angle, Amott imbibition, USBM displacement, and NMR technology are commonly used. NMR is more suitable for field and laboratory measurements of shale wettability due to its advantages of being rapid, non-destructive, and cost-effective [126, 127]. Since magnetic relaxation rates are positively correlated with fluid-wet surface interactions [128], NMR can be used to quantitatively measure rock wettability using the wettability index (WI).

WI is defined as the difference between the relative fractions of water-wet and oil-wet surfaces (Equation 25) [129, 130]. By introducing WI, Chen et al. correlated rock wettability with NMR surface relaxation behavior in Berea cores [131]. For unconventional shale, the combination of NMR and (oil and brine) imbibition experiments has successfully enabled qualitative wettability analysis [132, 133]. By modifying Equation (25) to (26), WI was further quantified, allowing for explicit quantification of reservoir rock wettability [105]. Additionally, for shale with significant residual fluids (e.g., Eagle Ford shale), Gupta et al. proposed a revised formula (Equation 28) to accommodate specific rock and fluid conditions [134].

$$ WI = \frac{A_w - A_o}{A_w + A_o} \tag{26} $$

$$ WI = \frac{\text{NMR}(S_w) - \text{NMR}(S_o)}{\text{NMR}(S_w) + \text{NMR}(S_o)} \tag{27} $$

$$ WI = \frac{[\text{NMR}(S_w) + \text{NMR}(S_{w,\text{in situ}})] - [\text{NMR}(S_o) + \text{NMR}(S_{o,\text{in situ}})]}{[\text{NMR}(S_w) + \text{NMR}(S_{w,\text{in situ}})] + [\text{NMR}(S_o) + \text{NMR}(S_{o,\text{in situ}})]} \tag{28} $$

where $WI$ is the wetting index, ranging from $-1$ (pure oil wetting) to $1$ (pure water wetting). From $0$ to $1$, $-0$, $0$ to $-1$, respectively, represent wetting by water, moderate wetting, and wetting by oil. A value of $0$ represents moderate wetting (or neutral wetting), indicating that neither fluid has a significant tendency toward the rock. A value between $-0.5$ and $0.5$ indicates conditions of mixed wetting at the pore scale (e.g., smaller pores are wetted by water, while larger pores are wetted by oil). Fractional wettability can also be obtained when there is significant heterogeneity in the dispersed region (e.g., the dispersed region is strongly oil-wet, while the rest of the region is water-wet) [130]. $A_w$ and $A_o$ are the surface areas wetted by water and oil, respectively; $\text{NMR}(S_w)$ and $\text{NMR}(S_o)$ are the saltwater NMR $T_2$ cumulative response and the ratio of the helium porosity of the sample to that of the adsorbed dodecane; $\text{NMR}(S_{w,\text{in situ}})$ and $\text{NMR}(S_{o,\text{in situ}})$ represent the NMR response of residual saltwater and hydrocarbons, respectively. Note that shale wettability exhibits great heterogeneity on a core scale [51].

*Table 4 summarizes various NMR methods for characterizing shale wettability, revealing the advantages and deficiencies of each method in terms of resolution capability, sensitivity to physical mechanisms, and applicable scenarios. The methods mentioned in the table are all low-field NMR techniques, whose experimental results are comparable to NMR logging data, and the lower magnetic field strength can limit the magnitude of internal magnetic field gradients within the reservoir, obtaining more accurate relaxation data.*

However, several deficiencies exist in published papers regarding NMR characterization of shale reservoir wettability:

1. The wettability relaxation mechanisms based on NMR remain unclear;
2. The influence of different NMR frequencies has not been considered when using NMR for wettability characterization;
3. There is a lack of complementary experimental support for NMR-based wettability characterization;
4. The frequency conversion of rock physics parameters related to the influence of local wettability on NMR response has not been quantified.

Additionally, current NMR instruments still struggle to detect relaxation signals from all solid components, and the application of non-steady-state diffusion in shale is limited. The selection of NMR wettability methods depends critically on a clear understanding of shale relaxation characteristics, requiring targeted design and optimization in terms of instrumentation and pulse sequences.

### 4.7 Dynamic Visualization of Nonuniform Pore Multiphase Seepage in Shale Based on Magnetic Resonance Imaging (MRI)

$^1$H MRI is a visualization method based on the NMR principle. The MRI technique is equipped with stronger magnetic field gradient pulses in addition to the static magnetic field. The additional RF gradient field is used to encode the spatial signals, enabling the visualization of spatial images of inhomogeneous and time-varying fluid flows and distributions in the shale, transforming the "black box" into a "white box" [146-150]. Currently, MRI has been successfully applied to study the interactions and behaviors on pore surfaces during fluid replacement in cores, as well as to study the dynamic distribution of multiphase fluids during water and CO₂ replacement (Figure 8A) [149, 151-153].

![Figure 8](placeholder_for_figure8.png)
**Figure 8:** (A) Low-field NMR rock core analyzer [154]. (B) (a) T1 encoding magnetization preparation with centric-scan SPRITE readout; (b) 3D superimposed images of oil and water in shale samples with cross-sections showing oil and water content; (c) T1 resolved 1D profiles for four shale samples; and (d) 2D cross-sections are from oil-suppressed 3D centric-scan SPRITE measurements perpendicular to the beddings of the shale sample with time during the water uptake experiment [43]. (C) ZTE imaging for high-resolution characterization of the shale pore structure and fluid distribution [155].

Zamiri et al. were the first to apply SPRITE MRI methods to give core plug size shale images. Water uptake experiments monitored using the MRI and $T_1 - T_2^*$ relaxation correlation measurements proved the capability of the $T_1 - T_2^*$ measurement to differentiate shale fracture water and pore water and demonstrated the key role of wettability in determining water spatial distribution in shales (Figure 8B) [43].

Recently, Ma et al. successfully implemented Zero Echo Time imaging (ZTE) technique under low-field NMR conditions for the first time, and the ZTE sequence is combined with relaxation NMR to obtain local information for shale samples before and after fluid self-absorption (Figure 8C) [155]. The results demonstrated that the ZTE technique enables the acquisition of high-quality shale images, effectively revealing sample heterogeneity and characterizing internal pore structures and fluid distributions at both macroscopic and microscopic scales. This method provides a reliable experimental approach for shale characterization and holds promise for broad application in the petroleum industry.

Combined with NMR relaxometry, the spatial distribution changes of NMR signals before and after displacement can be used to study fluid morphology and migration patterns. Lang et al. [156] studied the factors influencing CO₂ flooding in shale reservoirs using NMR and MRI and analyzed the impact of fracture development degree on CO₂ flooding in shale reservoirs (Figure 9A). They found that fractures enhance the contact area between injected CO₂ and crude oil and concluded that a higher degree of fracture development, along with an increased fracture evaluation index, leads to greater shale oil recovery [156].

![Figure 9](placeholder_for_figure9.png)
**Figure 9:** (A) MRI experimental study of CO2 injection to enhance shale oil recovery: (a) MRI of matrix core 5-1-26 at different times of CO2 injection; (b) MRI of the matrix core 5-20-27-2 with one vertical single fracture at different times of CO2 flooding; and (c) MRI of matrix core 5-18-27-1 with multiple high-angle fractures at different times of CO2 injection [156]. (B) (a) Schematics of EOR mechanisms for different aqueous phases with sc-CO2; and (b) MRI results of different core samples at different soaking stages [157]. (C) Shale oil reservoir production characteristics in microscopic pores developed by water/CO2 Huff-n-Puff [158]. (D) The degree of oil accumulation and its two-dimensional spatial distribution in the pore-fracture structures of interbedded shale during the charging process [159].

Yuan et al. [157] monitored the different stages of rock core samples from terrestrial shale immersed in supercritical carbon dioxide (sc-CO₂) and diluted microemulsion (DME) using MRI (Figure 9B). They showed that DME enables balanced mobilization of both heavy and light components, while sc-CO₂ enhances oil mobilization from the unswept area by the aqueous phase [157]. Therefore, combining sc-CO₂ and DME can result in a higher ultimate oil recovery factor in shale oil reservoirs.

Xie et al. [158] investigated the effects of deuterium water huff-n-puff (D₂O HnP), CO₂ HnP, and CO₂ HnP after D₂O HnP (CAD HnP) on crude oil recovery characteristics from micronanopores in shale through NMR relaxometry and MRI monitoring of shale spontaneous imbibition experiments (Figure 9C). They found that different medium HnP methods exhibited varying enhanced oil recovery (EOR) effects, with CAD HnP > CO₂ HnP > D₂O HnP [158].

Zhang et al. [159] combined MRI technology with oil injection and water flooding experiments to innovatively visualize and quantitatively characterize the dynamic micro-migration and mobility of hydrocarbons within the pore-fracture structure of shale (Figure 9D). The results showed that oil saturation significantly increased during the oil injection process, with the oil phase primarily accumulating in larger pores ($T_2 = 1 - 100$ ms) [159].

These studies collectively demonstrate that large pores and microfractures in shale significantly enhance the mobility of hydrocarbons within the matrix. Conversely, smaller pores contributed less to shale oil mobility due to higher capillary pressure ($P_c$) and flow resistance ($P_v$). Additionally, the efficiency of shale oil recovery enhancement varies markedly among different Huff-n-Puff media within shale micro-nanopores.

The combination of NMR relaxometry with MRI provides valuable methods for studying fluid behavior in shale, contributing to a deeper understanding of fluid occurrence and flow mechanisms in shale reservoirs. However, MRI instruments and pulse sequences suitable for high-resolution imaging of shale remain relatively scarce (Table 5). The inherently tight and heterogeneous pore structure of shale, combined with its enrichment in paramagnetic minerals, significantly enhances background noise and accelerates relaxation decay, thereby limiting the spatial resolution of MRI. Furthermore, fluid signals in micropores of ultra-low permeability and porosity reservoirs are prone to loss if they exceed the sampling range or are masked by noise. This means that details of these micropores and throats are overlooked, which is one of the reasons for low resolution.

*Table 5: MRI instruments and pulse sequences for shale research. (Refer to original PDF for the detailed table)*

## 5. Conclusions and Future Perspectives

NMR, as an innovative technology for characterizing unconventional shale reservoirs, has demonstrated unique application value through its relaxation spectrum analysis methods. This paper systematically introduces the theoretical foundations of NMR and a comprehensive analysis of its experimental techniques and analysis methods in shale research. However, despite the breakthrough discoveries in recent years, several technical challenges remain in practical applications.

In terms of experimental techniques, the NMR detection of shale fluids faces three core issues due to the limitations of existing instrument performance:

1. **Low signal-to-noise ratio and signal loss:** The $T_2$ relaxation of fluids in nanoscale pores approaches the instrument's detection limit (typically $< 1$ ms), making it difficult to effectively separate fluid and solid proton signals [22];
2. **Instrument probe dead time** (usually $>10 \, \mu\text{s}$) restricts the capture of ultra-short relaxation components, while the sensitivity of the RF coil directly impacts the comprehensive observation of the relaxation process [160];
3. **In time-domain NMR signal processing**, the lack of uniformity in ILT algorithms complicates peak identification, while traditional algorithms easily introduce artifacts [21]. To address these issues, new pulse sequences and optimized acquisition schemes need to be developed, along with standardized signal processing workflows, such as using signal differentiation methods to eliminate interference from dominant relaxation signals [161].

At the theoretical level, there are still key scientific gaps in understanding the relaxation mechanisms of the complex multiphase system of shale:

1. **Significant coupling effects of organic matter-clay-pyrite:** The dipole-dipole coupling of hydrogen nuclei within the organic matter dominates fluid signal modulation [162];
2. **The type of clay minerals** alters NMR responses by affecting pore wettability and moisture adsorption characteristics [163];
3. **Fe³⁺ ions** predominantly reside within the pyrite lattice rather than the pore surface, leading to limited paramagnetic relaxation contributions [164, 165]. These findings challenge the traditional hypothesis of surface relaxation dominance, highlighting the urgent need to develop new theoretical models that incorporate multi-physics field coupling.

At the technical application level, there are dual obstacles:

1. Existing experimental schemes, due to the complexity of RF pulse sequences, result in sample temperature rise, which contradicts the isothermal conditions required for reservoir conditions. This can be improved by optimizing pulse wait times [166] or incorporating online cooling systems;
2. The spatial resolution of MRI for fluid in nanopores is insufficient, primarily due to signal attenuation and signal overlap effects. It is also noteworthy that there remains a significant scale gap between the mechanistic understanding gained at the laboratory scale and its application in reservoir engineering.

**Future research should focus on:**

1. **Innovation in NMR experimental design and methods:** Such as the use of high-field and solid-state NMR, and the development of multinuclear NMR fixed-domain imaging measurements;
2. **Developing multi-parameter joint artificial intelligence inversion algorithms** that integrate relaxation-diffusion multi-dimensional information;
3. **Constructing a shale NMR "digital twin" database** to promote intelligent interpretation driven by big data.

It is foreseeable that nuclear magnetic resonance will become more interdisciplinary, and shale analysis will become a "big data" project, where laboratory experimental design should consider addressing unique engineering challenges.

## Author Contributions

**Long Zhou:** conceptualization, writing - original draft. **Guangzhi Liao:** conceptualization; writing - original draft, writing - review and editing, project administration, funding acquisition. **Ruiqi Fan:** conceptualization. **Rui Mao:** resources. **Xueli Hou:** investigation. **Nan Li:** supervision. **Zhilong He:** formal analysis. **Yushu Zhang:** visualization. **Lizhi Xiao:** writing - review and editing.

## References

[1] X. Jiang, X. Han, and Z. Cui, “New Technology for the Comprehensive Utilization of Chinese Oil Shale Resources,” *Energy* 32 (2007): 772–777.

[2] T. A. Blasingame, “The Characteristic Flow Behavior of Low‐Permeability Reservoir Systems,” paper presented at the SPE Unconventional Reservoirs Conference, Keystone, Colorado, USA, February 2008.

[3] C. Zou, D. Dong, S. Wang, et al., “Geological Characteristics and Resource Potential of Shale Gas in China,” *Petroleum Exploration and Development* 37 (2010): 641–653.

[4] Q. Zhang, L. Feng, Z. Pang, et al., “Reservoir Heterogeneity of the Longmaxi Formation and Its Significance for Shale Gas Enrichment,” *Energy Science & Engineering* 8, no. 12 (2020): 4229–4249.

[5] C. Kim and D. Devegowda, “Molecular Dynamics Study of Fluid‐Fluid and Solid‐Fluid Interactions in Mixed‐Wet Shale Pores,” *Fuel* 319 (2022): 123587.

[6] T. Wu, Z. Pan, B. Liu, L. D. Connell, R. Sander, and X. Fu, “Laboratory Characterization of Shale Oil Storage Behavior: A Comprehensive Review,” *Energy & Fuels* 35 (2021): 7305–7318.

[7] F. Hao, H. Zou, and Y. Lu, “Mechanisms of Shale Gas Storage: Implications for Shale Gas Exploration in China,” *AAPG Bulletin* 97 (2013): 1325–1346.

[8] Z. Jin, “Hydrocarbon Accumulation and Resources Evaluation: Recent Advances and Current Challenges,” *Advances in Geo‐Energy Research* 8 (2023): 1–4.

[9] M. Josh, L. Esteban, C. Delle Piane, J. Sarout, D. N. Dewhurst, and M. B. Clennell, “Laboratory Characterisation of Shale Properties,” *Journal of Petroleum Science and Engineering* 88–89 (2012): 107–124.

[10] J. M. Jin, S. Kim, and J. E. Birdwell, “Molecular Characterization and Comparison of Shale Oils Generated by Different Pyrolysis Methods,” *Energy & Fuels* 26 (2012): 1054–1062.

[11] B. Hazra, A. K. Varma, A. K. Bandopadhyay, et al., “FTIR, XRF, XRD and SEM Characteristics of Permian Shales,” *Journal of Natural Gas Science and Engineering* 32 (2016): 239–255.

[12] J. Zhang and G. Cheng, “Technical Aspects of the Pore Structure in Shale Measured by Small‐Angle and Ultrasmall‐Angle Neutron Scattering: A Mini Review,” *Energy & Fuels* 35 (2021): 1957–1965.

[13] Y. Yu, X. Luo, Z. Wang, et al., “A New Correction Method for Mercury Injection Capillary Pressure (MICP) to Characterize the Pore Structure of Shale,” *Journal of Natural Gas Science and Engineering* 68 (2019): 102896.

[14] B. Hazra, D. A. Wood, V. Vishal, and A. K. Singh, “Pore Characteristics of Distinct Thermally Mature Shales: Influence of Particle Size on Low‐Pressure CO2 and N2 Adsorption,” *Energy & Fuels* 32 (2018): 8175–8186.

[15] K. Wu, Z. Chen, J. Li, X. Li, J. Xu, and X. Dong, “Wettability Effect on Nanoconfined Water Flow,” *Proceedings of the National Academy of Sciences* 114 (2017): 3358–3363.

[16] S. Bakhshian, S. A. Hosseini, and N. Shokri, “Pore‐Scale Characteristics of Multiphase Flow in Heterogeneous Porous Media Using the Lattice Boltzmann Method,” *Scientific Reports* 9 (2019): 3377.

[17] U. Kuila, “Measurement and Interpretation of Porosity and Pore‐Size Distribution in Mudrocks: The Hole Story of Shales” (Doctoral dissertation, Colorado School of Mines, 2013).

[18] H. Pape, J. Arnold, R. Pechnig, et al., “Permeability Prediction for Low Porosity Rocks by Mobile NMR,” *Pure and Applied Geophysics* 166 (2009): 1125–1163.

[19] Y. Q. Song and R. Kausik, “NMR Application in Unconventional Shale Reservoirs—A New Porous Media Research Frontier,” *Progress in Nuclear Magnetic Resonance Spectroscopy* 112–113 (2019): 17–33.

[20] F. Bloch, “Nuclear Induction,” *Physical Review* 70 (1946): 460–474.

[21] V. V. Telkki, “Hyperpolarized Laplace NMR,” *Magnetic Resonance in Chemistry* 56 (2018): 619–632.

[22] R. Kausik, K. Fellah, E. Rylander, P. M. Singer, R. E. Lewis, and S. M. Sinclair, “NMR Relaxometry in Shale and Implications for Logging,” *Petrophysics* 57 (2016): 339–350.

[23] Z. Sun, M. Li, S. Yuan, et al., “The Flooding Mechanism and Oil Recovery of Nanoemulsion on the Fractured/Non‐Fractured Tight Sandstone Based on Online LF‐NMR Experiments,” *Energy* 291 (2024): 130226.

[24] V. Mlynárik, “Introduction to Nuclear Magnetic Resonance,” *Analytical Biochemistry* 529 (2017): 4–9.

[25] T. Hiller and N. Klitzsch, “Joint Inversion of Nuclear Magnetic Resonance Data From Partially Saturated Rocks Using a Triangular Pore Model,” *Geophysics* 83 (2018): JM15–JM28.

[26] V. V. Telkki and V. V. Zhivonitko, “Ultrafast NMR Diffusion and Relaxation Studies,” *Annual Reports on NMR Spectroscopy* 97 (2019): 83–119.

[27] M. Fleury and M. Romero‐Sarmiento, “Characterization of Shales Using T1–T2 NMR Maps,” *Journal of Petroleum Science and Engineering* 137 (2016): 55–62.

[28] E. T. Montrazi, E. Lucas‐Oliveira, A. G. Araujo‐Ferreira, M. Barsi‐Andreeta, and T. J. Bonagamba, “Simultaneous Acquisition for T2 – T2 Exchange and T1 – T2 Correlation NMR Experiments,” *Journal of Magnetic Resonance* 289 (2018): 63–71.

[29] J. Liu, Y. Fan, T. Qiu, X. Ge, S. Deng, and D. Xing, “A Novel Pulse Sequence and Inversion Algorithm of Three‐Dimensional Low Field NMR Technique in Unconventional Resources,” *Journal of Magnetic Resonance* 303 (2019): 67–74.

[30] J. Guo, M. S. Zamiri, and B. J. Balcom, “Optimization of Two‐Dimensional T1*–T2* Relaxation Correlation Measurements in Shale,” *Journal of Petroleum Science and Engineering* 217 (2022): 110939.

[31] Y. Q. Song, L. Venkataramanan, M. D. Hürlimann, M. Flaum, P. Frulla, and C. Straley, “T1–T2 Correlation Spectra Obtained Using a Fast Two‐Dimensional Laplace Inversion,” *Journal of Magnetic Resonance* 154 (2002): 261–268.

[32] A. Abragam, *The Principles of Nuclear Magnetism* (Oxford University Press, 1961).

[33] B. Zhou, “The Applications of NMR Relaxometry, NMR Cryoporometry, and FFC NMR to Nanoporous Structures and Dynamics in Shale at Low Magnetic Fields,” *Energy & Fuels* 32 (2018): 8897–8904.

[34] J. Mitchell, S. C. Stark, and J. H. Strange, “Probing Surface Interactions by Combining NMR Cryoporometry and NMR Relaxometry,” *Journal of Physics D: Applied Physics* 38 (2005): 1950–1958.

[35] M. Fleury, E. Kohler, F. Norrant, S. Gautier, J. M'Hamdi, and L. Barré, “Characterization and Quantification of Water in Smectites With Low‐Field NMR,” *Journal of Physical Chemistry C* 117 (2013): 4551–4560.

[36] J. E. Birdwell and K. E. Washburn, “Multivariate Analysis Relating Oil Shale Geochemical Properties to NMR Relaxometry,” *Energy & Fuels* 29 (2015): 2234–2243.

[37] M. Mehana and I. El‐monier, “Shale Characteristics Impact on Nuclear Magnetic Resonance (NMR) Fluid Typing Methods and Correlations,” *Petroleum* 2 (2016): 138–147.

[38] C. Guerrero and J. C. Santamarina, “Assessment of Hydrogen Adsorption in High Specific Surface Geomaterials Using Low‐Field NMR‐Implications for Storage and Field Characterization,” *International Journal of Hydrogen Energy* 95 (2024): 417–426.

[39] A. Borysenko, B. Clennell, I. Burgar, D. Dewhurst, R. Sedev, and J. Ralston, “Application of Low Field and Solid‐State NMR Spectroscopy to Study the Liquid/Liquid Interface in Porous Space of Clay Minerals and Shales,” *Diffusion Fundamentals* 10 (2009): 1–4, https://doi.org/10.62721/diffusion-fundamentals.10.422.

[40] J. C. Guo, H. Y. Zhou, J. Zeng, K. J. Wang, J. Lai, and Y. X. Liu, “Advances in Low‐Field Nuclear Magnetic Resonance (NMR) Technologies Applied for Characterization of Pore Space Inside Rocks: A Critical Review,” *Petroleum Science* 17 (2020): 1281–1297.

[41] M. S. Zamiri, B. MacMillan, F. Marica, J. Guo, L. Romero‐Zerón, and B. J. Balcom, “Petrophysical and Geochemical Evaluation of Shales Using Magnetic Resonance T1–T2* Relaxation Correlation,” *Fuel* 284 (2021): 119014.

[42] C. Xu, R. Xie, J. Guo, and J. Liu, “Wettability and Fluid Characterization in Shale Based on T1/T2 Variations in Solvent Extraction Experiments,” *Fuel* 355 (2024): 129512.

[43] M. S. Zamiri, F. Marica, L. Romero‐Zerón, and B. J. Balcom, “Monitoring Shale Water Uptake Using 2D Magnetic Resonance Relaxation Correlation and Sprite MRI,” *Chemical Engineering Journal* 428 (2022): 131042.

[44] Y. Liu, Y. Yao, D. Liu, S. Zheng, G. Sun, and Y. Chang, “Shale Pore Size Classification: An NMR Fluid Typing Method,” *Marine and Petroleum Geology* 96 (2018): 591–601.

[45] Y. Yuan and R. Rezaee, “Fractal Analysis of the Pore Structure for Clay Bound Water and Potential Gas Storage in Shales Based on NMR and N2 Gas Adsorption,” *Journal of Petroleum Science and Engineering* 177 (2019): 756–765.

[46] Z. H. Xie, “Clay Structure Water and Clay Bound Water Measured by NMR Relaxometry,” paper presented at the SPE/AAPG/SEG Unconventional Resources Technology Conference, Houston, Texas, USA, June, 2024.

[47] W. Yan, J. Sun, Z. Cheng, et al., “Petrophysical Characterization of Tight Oil Formations Using 1D and 2D NMR,” *Fuel* 206 (2017): 89–98.

[48] Z. Yang, Z. Ma, Y. Luo, Y. Zhang, H. Guo, and W. Lin, “A Measured Method for In Situ Viscosity of Fluid in Porous Media by Nuclear Magnetic Resonance,” *Geofluids* 2018 (2018): 1–8.

[49] M. D. Hürlimann, D. E. Freed, L. J. Zielinski, et al., “Hydrocarbon Composition From NMR Diffusion and Relaxation Data,” *Petrophysics* 50 (2009): 116–129.

[50] C. Zhang, F. Jiang, T. Hu, et al., “Oil Occurrence State and Quantity in Alkaline Lacustrine Shale Using a High‐Frequency NMR Technique,” *Marine and Petroleum Geology* 154 (2023): 106302.

[51] S. Wang, Z. Gu, P. Guo, and W. Zhao, “Comparative Laboratory Wettability Study of Sandstone, Tuff, and Shale Using 12‐MHz NMR T1‐T2 Fluid Typing: Insight of Shale,” *SPE Journal* 29 (2024): 4781–4803.

[52] P. M. Singer, Z. Chen, and G. J. Hirasaki, “Fluid Typing and Pore Size in Organic Shale Using 2D NMR in Saturated Kerogen Isolates,” *Petrophysics* 57 (2016): 604–619.

[53] S. Khatibi, M. Ostadhassan, Z. H. Xie, et al., “NMR Relaxometry a New Approach to Detect Geochemical Properties of Organic Matter in Tight Shales,” *Fuel* 235 (2019): 167–177.

[54] P. F. Zhang, S. F. Lu, J. J. Wang, et al., “Microscopic Occurrence and Distribution of Oil and Water In Situ Shale: Evidence From Nuclear Magnetic Resonance,” *Petroleum Science* 21 (2024): 3675–3691.

[55] J. Wang, S. Lu, P. Zhang, et al., “Characterization of Shale Oil and Water Micro‐Occurrence Based on a Novel Method for Fluid Identification by NMR T2 Spectrum,” *Fuel* 374 (2024): 132426.

[56] Y. Chi, B. Liu, L. Bai, Y. Su, Y. Huo, and E. Mohammadian, “Determination of Oil Contents in Ultralow Temperature Samples in High Maturity Shales Using 2D NMR Fluid Evaluation Technology,” *Marine and Petroleum Geology* 174 (2025): 107304.

[57] F. Panattoni, A. A. Colbourne, E. J. Fordham, J. Mitchell, C. P. Grey, and P. C. M. M. Magusin, “Improved Description of Organic Matter in Shales by Enhanced Solid Fraction Detection With Low‐Field 1H NMR Relaxometry,” *Energy & Fuels* 35 (2021): 18194–18209.

[58] Y. Zhang, D. Lv, Y. Wang, H. Liu, G. Song, and J. Gao, “Geological Characteristics and Abnormal Pore Pressure Prediction in Shale Oil Formations of the Dongying Depression, China,” *Energy Science & Engineering* 8 (2020): 1962–1979.

[59] B. Liu, X. W. Jiang, L. H. Bai, and R. S. Lu, “Investigation of Oil and Water Migrations in Lacustrine Oil Shales Using 20 MHz 2D NMR Relaxometry Techniques,” *Petroleum Science* 19 (2022): 1007–1018.

[60] Z. Lin, Q. Hu, N. Yin, S. Yang, H. Liu, and J. Chao, “Nanopores‐to‐Microfractures Flow Mechanism and Remaining Distribution of Shale Oil During Dynamic Water Spontaneous Imbibition Studied by NMR,” *Geoenergy Science and Engineering* 241 (2024): 213202.

[61] B. Sun, X. Liu, X. Zhao, et al., “Laminated Shale Oil System Mobility and Controlling Factors of the Paleogene Shahejie Formation: Evidences From T1‐T2 NMR Experiments, Multi‐Temperature Pyrolysis and Confocal Laser Scanning Microscopy,” *Fuel* 379 (2025): 133015.

[62] G. Li, X. Jin, Y. Shen, et al., “Comprehensive Evaluation of Microscopic Movability and Macroscopic Productivity of Continental Shale Reservoir,” *Journal of Energy Resources Technology, Part B: Subsurface Energy and Carbon Capture* 1 (2025): 011010.

[63] J. Lu, M. Liu, K. Liu, and Y. Zou, “A Novel Method for Quantitative Evaluation of Oil Content and Mobility in Shale Oil Reservoirs by NMR Logging: A Case Study of the Inter‐Salt Shale Oil in the Jianghan Basin,” *Unconventional Resources* 4 (2024): 100067.

[64] W. Ozowe, R. Russell, and M. Sharma, “A Novel Experimental Approach for Dynamic Quantification of Liquid Saturation and Capillary Pressure in Shale,” paper presented at the SPE/AAPG/SEG Unconventional Resources Technology Conference, Virtual, July 2020.

[65] S. Zhang, M. Wang, X. Zhu, C. Li, J. Cai, and J. Yan, “Oil Saturation Quantitative Evaluation in Lacustrine Shale: Novel Insights From NMR T1‐T2 and Displacement Experiments,” *Fuel* 371 (2024): 132062.

[66] B. Nicot, N. Vorapalawut, B. Rousseau, L. F. Madariaga, G. Hamon, and J. P. Korb, “Estimating Saturations in Organic Shales Using 2D NMR,” *Petrophysics* 57 (2016): 19–29.

[67] M. Gu, R. Xie, and G. Jin, “A New Quantitative Evaluation Method for Fluid Constituents With NMR T1‐T2 Spectra in Shale Reservoirs,” *Journal of Natural Gas Science and Engineering* 99 (2022): 104412.

[68] J. Li, M. Wang, J. Fei, et al., “Determination of In Situ Hydrocarbon Contents in Shale Oil Plays. Part 2: Two‐Dimensional Nuclear Magnetic Resonance (2D NMR) as a Potential Approach to Characterize Preserved Cores,” *Marine and Petroleum Geology* 145 (2022): 105890.

[69] X. Bai, X. Wang, M. Wang, et al., “Occurrence Characteristics and Factors That Influence Shale Oil in the Jurassic Lianggaoshan Formation, Northeastern Sichuan Basin,” *Marine and Petroleum Geology* 171 (2025): 107197.

[70] D. Cui, H. Yin, Y. Liu, J. Li, S. Pan, and Q. Wang, “Effect of Final Pyrolysis Temperature on the Composition and Structure of Shale Oil: Synergistic Use of Multiple Analysis and Testing Methods,” *Energy* 252 (2022): 124062.

[71] W. Ma, J. Li, and M. Wang, “Determination of In Situ Hydrocarbon Contents in Shale Oil Plays: Part 3: Quantification of Light Hydrocarbon Evaporative Loss in Old Cores Based on Preserved Shales,” *Marine and Petroleum Geology* 160 (2024): 106574.

[72] A. J. Olaide, E. Olugbenga, and D. Abimbola, “A Review of the Application of Nuclear Magnetic Resonance in Petroleum Industry,” *International Journal of Geosciences* 11 (2020): 145–169.

[73] M. Sandor, Y. Cheng, and S. Chen, “Improved Correlations for Heavy‐Oil Viscosity Prediction With NMR,” *Journal of Petroleum Science and Engineering* 147 (2016): 416–426.

[74] C. Cao Minh, S. Crary, L. Zielinski, C. B. Liu, S. Jones, and S. Jacobsen, “2D‐NMR Applications in Unconventional Reservoirs,” paper presented at the SPE Canadian Unconventional Resources Conference, Calgary, Alberta, Canada, October 2012.

[75] J. Bryan, D. Moon, and A. Kantzas, “In Situ Viscosity of Oil Sands Using Low Field NMR,” *Journal of Canadian Petroleum Technology* 44 (2005): 23–30.

[76] A. Tinni, C. Sondergeld, and C. Rai, “NMR T1‐T2 Response of Moveable and Non‐moveable Fluids in Conventional and Unconventional Rocks,” paper presented at the International Symposium of the Society of Core Analysts, Avignon, France, September, 2014.

[77] A. Rabbani and M. Babaei, “Image‐Based Modeling of Carbon Storage in Fractured Organic‐Rich Shale With Deep Learning Acceleration,” *Fuel* 299 (2021): 120795.

[78] Z. Sun, Y. Ni, Y. Wang, et al., “Experimental Investigation of the Effects of Different Types of Fracturing Fluids on the Pore Structure Characteristics of Shale Reservoir Rocks,” *Energy Exploration & Exploitation* 38 (2020): 682–702.

[79] J. Chen, X. Pang, H. Pang, Z. Chen, and C. Jiang, “Hydrocarbon Evaporative Loss Evaluation of Lacustrine Shale Oil Based on Mass Balance Method: Permian Lucaogou Formation in Jimusaer Depression, Junggar Basin,” *Marine and Petroleum Geology* 91 (2018): 422–431.

[80] H. Xu, D. Tang, J. Zhao, and S. Li, “A Precise Measurement Method for Shale Porosity With Low‐Field Nuclear Magnetic Resonance: A Case Study of the Carboniferous–Permian Strata in the Linxing Area, Eastern Ordos Basin, China,” *Fuel* 143 (2015): 47–54.

[81] M. Tan, K. Mao, X. Song, X. Yang, and J. Xu, “NMR Petrophysical Interpretation Method of Gas Shale Based on Core NMR Experiment,” *Journal of Petroleum Science and Engineering* 136 (2015): 100–111.

[82] P. Zhang, S. Lu, J. Li, C. Chen, H. Xue, and J. Zhang, “Petrophysical Characterization of Oil‐Bearing Shales by Low‐Field Nuclear Magnetic Resonance (NMR),” *Marine and Petroleum Geology* 89 (2018): 775–785.

[83] J. Li, S. Lu, G. Chen, M. Wang, S. Tian, and Z. Guo, “A New Method for Measuring Shale Porosity With Low‐Field Nuclear Magnetic Resonance Considering Non‐Fluid Signals,” *Marine and Petroleum Geology* 102 (2019): 535–543.

[84] J. Li, S. Lu, C. Jiang, et al., “Characterization of Shale Pore Size Distribution by NMR Considering the Influence of Shale Skeleton Signals,” *Energy & Fuels* 33, no. 7 (2019): 6361–6372.

[85] B. Ma, Q. Hu, S. Yang, et al., “Multiple Approaches to Quantifying the Effective Porosity of Lacustrine Shale Oil Reservoirs in Bohai Bay Basin, East China,” *Geofluids* 2020 (2020): 1–13.

[86] X. Ma, H. Wang, S. Zhou, Z. Feng, H. Liu, and W. Guo, “Insights Into NMR Response Characteristics of Shales and Its Application in Shale Gas Reservoir Evaluation,” *Journal of Natural Gas Science and Engineering* 84 (2020): 103674.

[87] Y. Gao, M. Wang, Y. Li, Z. Jiang, Y. Deng, and J. Qin, “Multi‐Scale Pore Structure Characterization of Lacustrine Fine‐Grained Mixed Sedimentary Rocks and Its Controlling Factors: A Case Study of Lucaogou Formation in Jimusar Sag,” *Energy & Fuels* 37 (2023): 977–992.

[88] D. Li, X. Tang, H. Zhang, et al., “Pore Structure Characterization and Reservoir Evaluation of Lacustrine Organic‐Rich Shale Based on the Nuclear Magnetic Resonance Test,” *Energy & Fuels* 38 (2024): 22050–22068.

[89] G. Xie, J. Sheng, J. Wang, et al., “Porosity and Pore Structure Evolution During the Weathering of Black Shale,” *Science of the Total Environment* 937 (2024): 173533.

[90] P. Zhao, B. He, B. Zhang, and J. Liu, “Porosity of Gas Shale: Is the NMR‐Based Measurement Reliable?,” *Petroleum Science* 19 (2022): 509–517.

[91] K. J. Dunn, D. J. Bergman, and G. A. LaTorraca, *Nuclear Magnetic Resonance: Petrophysical and Logging Applications* (Elsevier, 2002).

[92] Y. Yuan, R. Rezaee, M. F. Zhou, and S. Iglauer, “A Comprehensive Review on Shale Studies With Emphasis on Nuclear Magnetic Resonance (NMR) Technique,” *Gas Science and Engineering* 120 (2023): 205163.

[93] Y. Yuan, R. Rezaee, M. Verrall, S. Y. Hu, J. Zou, and N. Testmanti, “Pore Characterization and Clay Bound Water Assessment in Shale With a Combination of NMR and Low‐Pressure Nitrogen Gas Adsorption,” *International Journal of Coal Geology* 194 (2018): 11–21.

[94] M. N. Testamanti, “Assessment of Fluid Transport Mechanisms in Shale Gas Reservoirs” (Doctoral dissertation, Curtin University, 2018).

[95] A. Li, W. Ding, R. Wang, et al., “Petrophysical Characterization of Shale Reservoir Based on Nuclear Magnetic Resonance (NMR) Experiment: A Case Study of Lower Cambrian Qiongzhusi Formation in Eastern Yunnan Province, South China,” *Journal of Natural Gas Science and Engineering* 37 (2017): 29–38.

[96] L. Wang, L. Xiao, Y. Zhang, G. Liao, L. Wang, and W. Yue, “An Improved NMR Permeability Model for Macromolecules Flowing in Porous Medium,” *Applied Magnetic Resonance* 50 (2019): 1099–1123.

[97] W. Xie, Q. Yin, L. Wu, F. Yang, J. Zhao, and G. Wang, “A New Nuclear Magnetic Resonance‐Based Permeability Model Based on Two Pore Structure Characterization Methods for Complex Pore Structure Rocks: Permeability Assessment in Nanpu Sag, China,” *Geophysics* 89 (2024): MR43–MR51.

[98] M. Saidian and M. Prasad, “Effect of Mineralogy on Nuclear Magnetic Resonance Surface Relaxivity: A Case Study of Middle Bakken and Three Forks Formations,” *Fuel* 161 (2015): 197–206.

[99] P. Zhao, L. Wang, C. Xu, et al., “Nuclear Magnetic Resonance Surface Relaxivity and Its Advanced Application in Calculating Pore Size Distributions,” *Marine and Petroleum Geology* 111 (2020): 66–74.

[100] Y. Yuan and R. Rezaee, “Impact of Paramagnetic Minerals on NMR‐Converted Pore Size Distributions in Permian Carynginia Shales,” *Energy & Fuels* 33 (2019): 2880–2887.

[101] H. Daigle, N. W. Hayman, E. D. Kelly, K. L. Milliken, and H. Jiang, “Fracture Capture of Organic Pores in Shales,” *Geophysical Research Letters* 44 (2017): 2167–2176.

[102] M. D. Hurlimann, K. G. Helmer, L. L. Latour, and C. H. Sotak, “Restricted Diffusion in Sedimentary Rocks. Determination of Surface‐Area‐to‐Volume Ratio and Surface Relaxivity,” *Journal of Magnetic Resonance, Series A* 111 (1994): 169–178.

[103] K. Keating and R. Knight, “A Laboratory Study of the Effect of Fe (II)‐Bearing Minerals on Nuclear Magnetic Resonance (NMR) Relaxation Measurements,” *Geophysics* 75 (2010): F71–F82.

[104] K. Keating and R. Knight, “The Effect of Spatial Variation in Surface Relaxivity on Nuclear Magnetic Resonance Relaxation Rates,” *Geophysics* 77 (2012): E365–E377.

[105] I. Sulucarnain, C. H. Sondergeld, and C. S. Rai, “An NMR Study of Shale Wettability and Effective Surface Relaxivity,” paper presented at the SPE Canadian Unconventional Resources Conference, Calgary, Alberta, Canada, October, 2012.

[106] S. Rivera, “Ultrasonic and Low Field Nuclear Magnetic Resonance Study of Lower Monterey Formation: San Joaquin Basin,” Colorado School of Mines, Arthur Lakes Library, 2014.

[107] L. O. Lawal, A. R. Adebayo, M. Mahmoud, B. M. Dia, and A. S. Sultan, “A Novel NMR Surface Relaxivity Measurements on Rock Cuttings for Conventional and Unconventional Reservoirs,” *International Journal of Coal Geology* 231 (2020): 103605.

[108] J. Mitchell, J. Webber, and J. Strange, “Nuclear Magnetic Resonance Cryoporometry,” *Physics Reports* 461 (2008): 1–36.

[109] O. V. Petrov and I. Furó, “NMR Cryoporometry: Principles, Applications and Potential,” *Progress in Nuclear Magnetic Resonance Spectroscopy* 54 (2009): 97–122.

[110] J. H. Strange, M. Rahman, and E. G. Smith, “Characterization of Porous Solids by NMR,” *Physical Review Letters* 71 (1993): 3589–3591.

[111] P. M. Kekkonen, A. Ylisassi, and V. V. Telkki, “Absorption of Water in Thermally Modified Pine Wood as Studied by Nuclear Magnetic Resonance,” *Journal of Physical Chemistry C* 118 (2014): 2146–2153.

[112] C. L. Jackson and G. B. McKenna, “The Melting Behavior of Organic Materials Confined in Porous Solids,” *Journal of Chemical Physics* 93 (1990): 9002–9011.

[113] S. Tong, Y. Dong, Q. Zhang, D. Elsworth, and S. Liu, “Quantitative Analysis of Nanopore Structural Characteristics of Lower Paleozoic Shale, Chongqing (Southwestern China): Combining FIB‐SEM and NMR Cryoporometry,” *Energy & Fuels* 31 (2017): 13317–13328.

[114] M. Fleury, R. Fabre, and J. B. W. Webber, “Comparison of Pore Size Distribution by NMR Relaxation and NMR Cryoporometry in Shales,” paper presented at the International Symposium of the Society of Core Analysts, St. John's, NL, Canada, August 2015.

[115] B. Zhou, Q. Han, and P. Yang, “Characterization of Nanoporous Systems in Gas Shales by Low Field NMR Cryoporometry,” *Energy & Fuels* 30 (2016): 9122–9131.

[116] Q. Zhang, Y. Dong, S. Liu, D. Elsworth, and Y. Zhao, “Shale Pore Characterization Using NMR Cryoporometry With Octamethylcyclotetrasiloxane as the Probe Liquid,” *Energy & Fuels* 31 (2017): 6951–6959.

[117] S. Iglauer, M. A. Fernø, P. Shearing, and M. J. Blunt, “Comparison of Residual Oil Cluster Size Distribution, Morphology and Saturation in Oil‐Wet and Water‐Wet Sandstone,” *Journal of Colloid and Interface Science* 375 (2012): 187–192.

[118] S. Iglauer, “CO2–Water–Rock Wettability: Variability, Influencing Factors, and Implications for CO2 Geostorage,” *Accounts of Chemical Research* 50 (2017): 1134–1142.

[119] Y. Guo, X. Li, and L. Huang, “Insight Into Spontaneous Water‐Based Working Fluid Imbibition on the Dynamic Tensile Behavior of Anisotropic Shale,” *Engineering Geology* 308 (2022): 106830.

[120] B. Pan, X. Yin, W. Zhu, et al., “Theoretical Study of Brine Secondary Imbibition in Sandstone Reservoirs: Implications for H2, CH4 , and CO2 Geo‐storage,” *International Journal of Hydrogen Energy* 47 (2022): 18058–18066.

[121] A. Valori and B. Nicot, “A Review of 60 Years of NMR Wettability,” *Petrophysics—The SPWLA Journal of Formation Evaluation and Reservoir Description* 60 (2019): 255–263.

[122] W. Anderson, “Wettability Literature Survey—Part 2: Wettability Measurement,” *Journal of Petroleum Technology* 38 (1986): 1246–1262.

[123] M. A. Q. Siddiqui, S. Ali, H. Fei, and H. Roshan, “Current Understanding of Shale Wettability: A Review on Contact Angle Measurements,” *Earth‐Science Reviews* 181 (2018): 1–11.

[124] H. Roshan, A. Z. Al‐Yaseri, M. Sarmadivaleh, and S. Iglauer, “On Wettability of Shale Rocks,” *Journal of Colloid and Interface Science* 475 (2016): 104–111.

[125] S. Chen, P. Li, J. Zhang, et al., “Measurement of Shale Wettability Using Calorimetry: Experimental Results and Model,” *Energy & Fuels* 35 (2021): 17446–17462.

[126] M. Ali, A. Al‐Yaseri, F. U. R. Awan, M. Arif, A. Keshavarz, and S. Iglauer, “Effect of Water‐Soluble Organic Acids on Wettability of Sandstone Formations Using Streaming Zeta Potential and NMR Techniques: Implications for CO2 Geo‐Sequestration,” *Fuel* 329 (2022): 125449.

[127] Y. Yuan, R. Rezaee, J. Zou, and K. Liu, “Pore‐Scale Study of the Wetting Behavior in Shale, Isolated Kerogen, and Pure Clay,” *Energy & Fuels* 35 (2021): 18459–18466.

[128] R. J. Brown and I. Fatt, “Measurements of Fractional Wettability of Oil Fields' Rocks by the Nuclear Magnetic Relaxation Method,” paper presented at the Fall Meeting of the Petroleum Branch of AIME, Los Angeles, California, October, 1956.

[129] M. Fleury and F. Deflandre, “Quantitative Evaluation of Porous Media Wettability Using NMR Relaxometry,” *Magnetic Resonance Imaging* 21 (2003): 385–387.

[130] W. Looyestijn and J. Hofman, “Wettability‐Index Determination by Nuclear Magnetic Resonance,” *SPE Reservoir Evaluation & Engineering* 9 (2006): 146–153.

[131] J. Chen, G. J. Hirasaki, and M. Flaum, “NMR Wettability Indices: Effect of OBM on Wettability and NMR Responses,” *Journal of Petroleum Science and Engineering* 52 (2006): 161–171.

[132] E. Odusina, C. Sondergeld, and C. Rai, “An NMR Study on Shale Wettability,” paper presented at the Canadian Unconventional Resources Conference, Calgary, Alberta, Canada, November 2011.

[133] E. Odusina and R. F. Sigal, “Laboratory NMR Measurements on Methane Saturated Barnett Shale Samples,” *Petrophysics* 52 (2011): 32–49.

[134] I. Gupta, C. Rai, and C. Sondergeld, “Study Impact of Sample Treatment and In Situ Fluids on Shale Wettability Measurement Using NMR,” *Journal of Petroleum Science and Engineering* 176 (2019): 352–361.

[135] S. Su, Z. Jiang, X. Shan, et al., “The Wettability of Shale by NMR Measurements and Its Controlling Factors,” *Journal of Petroleum Science and Engineering* 169 (2018): 309–316.

[136] J. Liu and J. J. Sheng, “Experimental Investigation of Surfactant Enhanced Spontaneous Imbibition in Chinese Shale Oil Reservoirs Using NMR Tests,” *Journal of Industrial and Engineering Chemistry* 72 (2019): 414–422.

[137] L. Bai, B. Liu, Y. Huo, et al., “Liquid Spontaneous Imbibition and Its Time‐Resolved Nuclear Magnetic Resonance Within Differently Matured Shale Indications for Shale Pore Structure and Wettability,” *Energy & Fuels* 38 (2024): 22804–22819.

[138] Q. Lv, J. Hou, Z. Cheng, et al., “Wettability and Imbibition Mechanism of the Multiscale Pore Structure of Jiyang Shale Oil Formation by NMR,” *Energy & Fuels* 39 (2025): 3097–3109.

[139] M. J. Dick, D. Veselinovic, R. J. M. Bonnie, and S. A. Kelly, “NMR‐Based Wettability Index for Unconventional Rocks,” *Petrophysics–The SPWLA Journal of Formation Evaluation and Reservoir Description* 63 (2022): 418–441.

[140] L. Wang, R. Yin, L. Sima, et al., “Insights Into Pore Types and Wettability of a Shale Gas Reservoir by Nuclear Magnetic Resonance: Longmaxi Formation, Sichuan Basin, China,” *Energy & Fuels* 32 (2018): 9289–9303.

[141] J. Liu, R. Xie, and J. Guo, “Pore‐Scale T2‐Based Numerical Investigation on Dynamics and Wettability in Mixed‐Wet Shale Oil Reservoirs,” *Physics of Fluids* 36 (2024): 122025.

[142] D. Merkel, M. Stephans, C. Thompson, and K. McLean, “Log and Core NMR T1‐T2 and T2‐D Mapping of the Bakken Reservoir Complex,” paper presented at the Unconventional Resources Technology Conference, Houston, Texas, July 2021.

[143] J. P. Korb, B. Nicot, and I. Jolivet, “Dynamics and Wettability of Petroleum Fluids in Shale Oil Probed by 2D T1‐T2 and Fast Field Cycling NMR Relaxation,” *Microporous and Mesoporous Materials* 269 (2018): 7–11.

[144] J. P. Korb, B. Nicot, A. Louis‐Joseph, S. Bubici, and G. Ferrante, “Dynamics and Wettability of Oil and Water in Oil Shales,” *Journal of Physical Chemistry C* 118 (2014): 23212–23218.

[145] R. M. Steele, J. P. Korb, G. Ferrante, and S. Bubici, “New Applications and Perspectives of Fast Field Cycling NMR Relaxometry,” *Magnetic Resonance in Chemistry* 54 (2016): 502–509.

[146] J. Mitchell, T. C. Chandrasekera, D. J. Holland, L. F. Gladden, and E. J. Fordham, “Magnetic Resonance Imaging in Laboratory Petrophysical Core Analysis,” *Physics Reports* 526 (2013): 165–225.

[147] Y. Cheng, Q. Di, C. Gu, F. Ye, S. Hua, and P. Yang, “Visualization Study on Fluid Distribution and End Effects in Core Flow Experiments With Low‐Field MRI Method,” *Journal of Hydrodynamics* 27 (2015): 187–194.

[148] E. Lev and C. M. Boyce, “Opportunities for Characterizing Geological Flows Using Magnetic Resonance Imaging,” *iScience* 23 (2020): 101534.

[149] J. Siavashi, A. Najafi, M. Sharifi, et al., “An Insight Into Core Flooding Experiment via NMR Imaging and Numerical Simulation,” *Fuel* 318 (2022): 123589.

[150] Q. Chen, W. Kinzelbach, and S. Oswald, “Nuclear Magnetic Resonance Imaging for Studies of Flow and Transport in Porous Media,” *Journal of Environmental Quality* 31 (2002): 477–486.

[151] A. Afrough, M. Shakerian, M. S. Zamiri, et al., “Magnetic‐Resonance Imaging of High‐Pressure Carbon Dioxide Displacement: Fluid/Surface Interaction and Fluid Behavior,” *SPE Journal* 23 (2018): 772–787.

[152] Y. Zhao, Y. Zhang, X. Lei, Y. Zhang, and Y. Song, “CO2 Flooding Enhanced Oil Recovery Evaluated Using Magnetic Resonance Imaging Technique,” *Energy* 203 (2020): 117878.

[153] T. Zhang, M. Tang, Y. Ma, et al., “Experimental Study on CO2/Water Flooding Mechanism and Oil Recovery in Ultralow‐Permeability Sandstone With Online LF‐NMR,” *Energy* 252 (2022): 123948.

[154] N. Zhang, S. Wang, J. Wu, Z. Li, and F. Zhao, “Characterization of Pore Structure and Dynamic Seepage Characteristics of Sandstone Determined by Nuclear Magnetic Resonance (NMR) and Magnetic Resonance Imaging (MRI) Techniques,” *Journal of Porous Media* 27 (2024): 85–99.

[155] Z. Ma, Y. Zhang, and L. Xiao, “ZTE Imaging for High‐Resolution Characterization of the Shale Pore Structure and Fluid Distribution,” *Energy & Fuels* 39 (2025): 6208–6219.

[156] D. Lang, Z. Lun, C. Lyu, H. Wang, Q. Zhao, and H. Sheng, “Nuclear Magnetic Resonance Experimental Study of CO2 Injection to Enhance Shale Oil Recovery,” *Petroleum Exploration and Development* 48 (2021): 702–712.

[157] S. Yuan, B. Wang, M. Yang, et al., “Synergistic Effects Between Supercritical CO2 and Diluted Microemulsion on Enhanced Oil Recovery in Shale Oil Reservoirs,” *SPE Journal* 30 (2025): 295–309.

[158] Z. Xie, Y. Xiong, Z. Song, J. Chang, K. Zhang, and Z. Fan, “Shale Oil Reservoir Production Characteristics in Microscopic Pores Developed by Water/CO2 Huff‐n‐Puff,” *Energy & Fuels* 39 (2025): 3517–3527.

[159] Y. Zhang, J. Chang, Z. Jiang, et al., “Visualization of Dynamic Micro‐Migration of Shale Oil and Investigation of Shale Oil Movability by NMRi Combined Oil Charging/Water Flooding Experiments: A Novel Approach,” *Marine and Petroleum Geology* 165 (2024): 106907.

[160] Y. Q. Song, “Magnetic Resonance of Porous Media (MRPM): A Perspective,” *Journal of Magnetic Resonance* 229 (2013): 12–24.

[161] E. Müller‐Huber, F. Börner, J. H. Börner, and D. Kulke, “Combined Interpretation of NMR, MICP, and SIP Measurements on Mud‐Dominated and Grain‐Dominated Carbonate Rocks,” *Journal of Applied Geophysics* 159 (2018): 228–240.

[162] K. E. Washburn and J. E. Birdwell, “Updated Methodology for Nuclear Magnetic Resonance Characterization of Shales,” *Journal of Magnetic Resonance* 233 (2013): 17–28.

[163] H. Daigle, A. Johnson, J. P. Gips, and M. Sharma, “Porosity Evaluation of Shales Using NMR Secular Relaxation,” paper presented at the Unconventional Resources Technology Conference, Denver, Colorado, August 2014.

[164] Z. Jia, L. Xiao, Z. Wang, G. Liao, Y. Zhang, and C. Liang, “Molecular Dynamics and Composition of Crude Oil by Low‐Field Nuclear Magnetic Resonance,” *Magnetic Resonance in Chemistry* 54 (2016): 650–655.

[165] Z. Jia, L. Xiao, Z. Wang, et al., “Magic Echo for Nuclear Magnetic Resonance Characterization of Shales,” *Energy & Fuels* 31 (2017): 7824–7830.

[166] F. Zuo, X. Ge, L. Xing, et al., “An Improved Method to Accelerate the Acquisition Efficiency of the T1–T2 Spectrum Based on the IR‐BSSFP‐CPMG Pulse Sequence,” *Computers & Geosciences* 196 (2025): 105792.
