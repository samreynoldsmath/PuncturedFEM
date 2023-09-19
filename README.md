```
______                 _                      _______ ________  ___
| ___ \               | |                    | |  ___|  ___|  \/  |
| |_/ /   _ _ __   ___| |_ _   _ _ __ ___  __| | |_  | |__ |      |
|  __/ | | | '_ \ / __| __| | | | '__/ _ \/ _  |  _| |  __|| |\/| |
| |  | |_| | | | | (__| |_| |_| | | |  __/ (_| | |   | |___| |  | |
\_|   \____|_| |_|\___|\__|\____|_|  \___|\____\_|   \____/\_|  |_/
```
https://github.com/samreynoldsmath/PuncturedFEM

## Description
A finite element method on meshes with curvilinear and multiply connected cells.

## Examples used in publications
* "Evaluation of inner products of implicitly-defined finite element functions
  on multiply connected mesh cells," J. S. Ovall and S. E. Reynolds, in review.
  * [Example 4.1 (Punctured Square)](examples/ex1a-square-hole.ipynb)
  * [Example 4.2 (Pac-Man)](examples/ex1b-pacman.ipynb)
  * [Example 4.3 (Ghost)](examples/ex1c-ghost.ipynb)

## Dependencies
This project is written in Python 3.11 and uses the following packages:
* [matplotlib](https://matplotlib.org/)
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)
* [jupyter](https://jupyter.org/) (optional)
* [tqdm](https://tqdm.github.io/) (optional)
See [requirements.txt](requirements.txt) for a complete list of dependencies.

## Authors
[Jeffrey S. Ovall](https://sites.google.com/pdx.edu/jeffovall)
and
[Samuel E. Reynolds](https://sites.google.com/view/samreynolds)

## Affiliation
Fariborz Maseeh Department of Mathematics and Statistics<br>
Portland State University<br>
Portland, Oregon, USA

## Acknowledgements
Funding for this project was provided by the National Science Foundation through
**NSF grant DMS-2012285** and **NSF RTG grant DMS-2136228**.

## Disclaimers
* This code is intended to serve as a prototype, and has not necessarily been optimized for performance.
* This project is under heavy development, which may result to changes in to the API. Consult the examples for the latest suggested usage.

## License
Copyright (C) 2022 - 2023 Jeffrey S. Ovall and Samuel E. Reynolds.

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
