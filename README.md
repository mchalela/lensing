# Lensing: Package for weak lensing analysis

Lensing is a set of tools to work with lensing catalogs, measure density profiles and fit compound density models. The package is divided into several modules: LensCat, DensModel, Shear, Kappa and gentools.

- **LensCat**: Allows you to easily create source catalogues for your lenses. Right now the available lensing catalogues are KiDS, CS82, CFHTLens and RCSL. Each catalogue has different properties so the idea is to provide a unified interface and namespace for all of them. [Read more](LensCat/README.md)

- **Shear**: Provides a few methods to compute shear profiles and maps. [Read more](Shear/README.md)

- **Kappa**: Computes the kappa density map using the Kaiser-Squires inversion method. [Read more](Kappa/README.md)

- **DensModel**: Defines different mass models and fitting methods to fit the Shear profiles. [Read more](DensModel/README.md)

- **gentools**: This is a module for general tools that are used by the rest of the modules. [Read more](gentools/README.md)
