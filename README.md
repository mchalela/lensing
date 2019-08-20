# Lensing: Module for weak lensing analysis

Lensing is a set of tools to work with lensing catalogs, measure density profiles and fit compound density models. The module is divided into several submodules: lenscat, shear, densmodel, gentools and kappa (not implemented yet).

## Lenscat
Lenscat allows you to easily create source catalogues for your lenses. Right now the available lensing catalogues are KiDS, CS82, CFHTLens... Each catalogue has different properties so the idea is to provide a unified interface and namespace for all of them.
