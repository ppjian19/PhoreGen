# PhoreGen: Pharmacophore-Oriented 3D Molecular Generation towards Efficient Feature-Customized Drug Discovery

[PhoreGen](https://phoregen.ddtmlab.org) is a pharmacophore-oriented 3D molecular generation framework designed to generate entire 3D molecules that are precisely aligned with a given pharmacophore model. It employs asynchronous perturbations and simultaneously updates on both atomic and bond information, coupled with a message-passing mechanism that incoporates prior knowledge of ligand-pharmacophore mapping during the diffusion-denoising process. By hierarchical learning on a large number of ligand-pharmacophore pairs derived from 3D ligands, complex structures, and docking-produced potential binding modes, PhoreGen can generate chemically and energetically reasonable 3D molecules well-aligned with the pharmacophore constraints, while maintaining structural diversity, drug-likeness, and potentially high binding affinity. Notably, it excels in generating feature-customized molecules, e.g. with covalent groups and metal-binding motifs, at high frequency, demonstrating its unparalleled ability and practicality even for challenging drug design scenarios.

## Here shows the process of PhoreGen generating an entire 3D molecule under the pharmacophore constraint.
<img src="./assets/generation demo.gif" alt="model"  width="70%"/>

## Here shows an example of PhoreGen generating new molecules for metallo- and serine-β-lactamases.
<img src="./assets/MBL-SBL demo.gif" alt="model"  width="70%"/>

## How to create a pharmacophore model for PhoreGen application
We here provide a user-friendly web server for this purpose.

Please visit https://ancphore.ddtmlab.org/Modeling

## Contact

For questions or feedback, please contact:
- **Peng Jian**: ppjian19@163.com
- **Li Guo-Bo**: liguobo@scu.edu.cn
- Visit our [Lab Website](https://ddtmlab.org) for more details about PhoreGen and related projects.
