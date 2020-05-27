======
How To
======
This package is meant as a facility to ease the management of maps and profiles associated
with a galactic disc, and provides a few functions all related to dynamical quantities and
modelling. The package itself is quickly evolving.

pydisc is a python package meant to ease the use the manipulation of maps and profiles
and the computation of basic quantities pertaining to galactic Discs.

1 - Structures in pydisc
------------------------

DataMaps, DataProfiles, Maps and Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the language of pydisc:

- **DataMaps** are data on a grid (hence 2D), e.g., values like velocities, flux, etc. The grid on which it is defined should be regular.
- **DataProfiles** are data on a radial profile (hence 1D).
- **Maps** are then a set of DataMaps associated with a set of coordinates X, Y.
- **Profiles** are then a set of DataProfiles associated with a set of radial coordinates R.
- **DataMaps** have orientations defined as 'NE_direct' indicating if the North-to-East axis is direct (counter-clockwise) or indirect (clockwise). It also has an 'alpha_North' angle which provides the angle between the North and the top (positive y-axis). DataMaps also have a pixelscale which provides the conversion between arcseconds and pixels in case the X,Y grids are not defined. If this is the case, X, and Y will be computed using just the indices from the grid.

**DataMaps** and **DataProfiles** have 'units' as defined by astropy units. These should be compatible with e.g., arcseconds, so these are observational.

DataMap arguments:
""""""""""""""""""
- dunit: astropy unit
- order: velocity moment order. Hence velocities are order=1, flux or mass is 0, dispersion is 2, anything else would be -1 and there is a category for "dummy" maps with order=-10.
- dname: name of the datamap
- flag: a flag which is meant to add info (string)
- data and edata: numpy arrays. If edata is not provided, it will be defined as None.

**DataProfiles** have similar arguments, but with punit (profile unit) and pname.

Maps arguments:
"""""""""""""""

- name: name of the map
- X and Y: the 2 main arrays. If not provided, indices will be used.
- Xcen and Ycen: centre for the 0,0
- XYunit: unit (astropy) for the X and Y axis
- NE_direct, alpha_North, etc.

Note that a **Map** can have many **DataMaps**: hence a set of X,Y can have many data
associated to it (sharing the same coordinates), each one having a different dname,
order, flag etc.

Galaxy
""""""
A 'Galaxy' is an object which has some characteristics like: a distance, a Position Angle
for the line of Nodes, an inclination (in degrees) and the Position Angle for a bar if
present.

GalacticDisc
"""""""""""""
A 'GalacticDisc' is a structure associating a set of Maps and Profiles and a given Galaxy.

This is the main structure which we will be using for the calculation of various quantities.

There are a number of associated classes, namely:

DensityWave
"""""""""""""
This is associated with methods for density waves like the Tremaine Weinberg method

GalacticTorque
""""""""""""""
Associated with methods for deriving torques all inheriting from the GalacticDisc class, thus sharing a number of functionalities, but also have their own specific ones (which require a set of maps).

2- Grammar for Maps/Profiles
-------------------------------
The 'grammar' for maps and datamaps is simple (a priori): if you have an attribute like
"data" you can input this in the argument list as: "data_". Hence if the map is name "MUSE"
and the datamap named "vstar" you should have an argument for the data as "dataMUSE_vstar"
and the associated "edataMUSE_vstar" if you have uncertainties for this map etc.
Same applies for all argument of the maps and data, for example (using the same example):
orderMUSE_vstar, XMUSE, YMUSE, XcenMUSE, YcenMUSE, flagMUSE_vstar...
In this way you can have several datamaps attached to a single map and have e.g.,:
XMUSE, YMUSE, dataMUSE_vstar, dataMUSE_gas, dataMUSE_...