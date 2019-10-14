# Lensing: Package for weak lensing analysis

Lensing is a set of tools to work with lensing catalogs, measure density profiles and fit compound density models. The package is divided into several modules: LensCat, DensModel, Shear, Kappa and gentools.

- **LensCat**: Allows you to easily create source catalogues for your lenses. Right now the available lensing catalogues are KiDS, CS82, CFHTLens... Each catalogue has different properties so the idea is to provide a unified interface and namespace for all of them.

- **Shear**: Provides a few methods to compute shear profiles and maps.

- **Kappa**: Computes the kappa density map using the Kaiser-Squires inversion method.

- **DensModel**: Defines different mass models and fitting methods to fit the Shear profiles.

- **gentools**: This is a module for general tools that are used by the rest of the modules.
