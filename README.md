# Sparse Identification of Truncation Errors (SITE)

This project contains the implementation of SITE, an approach
to identify modified differential equations from simulation
data as shown in 'Sparse Identification of Truncation Errors' 
of Stephan Thaler, Ludger Paehler and Nikolaus A. Adams. 
All codes used to generate the data and the plots of the 
paper are given here. A detailed explanation of the approach 
can be found in the [preprint](https://www.google.com). 

## Built With

* [SymPy](https://www.sympy.org/en/index.html) - For the symbolic computations 
in the method of manufactured solutions and analytic derivations of modified differential equations (version 1.3)
* [geomdl](https://pypi.org/project/geomdl/) - For computations with NURBS (version 4.4.1)
* [PySwarms](https://pyswarms.readthedocs.io/en/latest/) - For the particle swarm 
optimization (version 0.3.1)
* [findiff](https://pypi.org/project/findiff/) - For finite difference approximations (version 0.6.2)
* [scikit-learn](https://scikit-learn.org/stable/) - For the implementation of Lasso (version 0.20.2)

We also used numpy (version 1.15.4), scipy (version 1.1.0) and matplotlib (version 3.0.2).

To run the SR3 sparse regression algorithm, the 
[MATLAB-Python interface](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html)
is used. If you want to run SR3, you have to set up the
MATLAB Engine API.

## Example

The functionality of the SITE approach can be controlled from within the
'main_file'. All controllable parameters are explained in this file as well.

## Updates

Feel free to submit bug-fix requests through
the issues tab on GitHub.

## Code Author

[**Stephan Thaler**](https://github.com/s-thaler)

Co-authors of the paper: [Ludger Paehler](https://ludgerpaehler.github.io/)
and [Nikolaus A. Adams](http://www.professoren.tum.de/en/adams-nikolaus/)


## License

This project is licensed under the MIT License - see 
the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Samuel Rudy for parts of the 
[PDE-FIND](https://github.com/snagcliffs/PDE-FIND) implementation
and the consent to publish these parts here 
* Travis Askham and Peng Zheng for the implementation of
[SR3](https://github.com/UW-AMO/sr3-matlab)



## Citation
```
@article{thaler2019site,
  title={Sparse Identification of Truncation Errors},
  author={Thaler, Stephan and Paehler, Ludger and Adams, Nikolaus A.},
  journal={arXiv preprint arXiv:},
  year={2019}
}
```