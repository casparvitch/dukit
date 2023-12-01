URLS=[
"dukit/index.html",
"dukit/field/index.html",
"dukit/field/defects.html",
"dukit/pl/index.html",
"dukit/pl/common.html",
"dukit/pl/interface.html",
"dukit/pl/model.html",
"dukit/pl/scipyfit.html",
"dukit/pl/cpufit.html",
"dukit/driftcorrect.html",
"dukit/magsim.html",
"dukit/rebin.html",
"dukit/geom.html",
"dukit/fourier.html",
"dukit/json2dict.html",
"dukit/share.html",
"dukit/warn.html",
"dukit/itool.html",
"dukit/polygon.html",
"dukit/systems.html",
"dukit/widget.html",
"dukit/plot.html"
];
INDEX=[
{
"ref":"dukit",
"url":0,
"doc":"Package for the analysis of (widefield) defect microscopy image data, from the Tetienne lab at RMIT/UniMelb. For the lab control software see [DavidBroadway/qdm-control](https: github.com/DavidBroadway/qdm-control) (may not be public, talk to David Broadway). On this page we will document the API, for other information see the [README.md](https: github.com/casparvitch/dukit/blob/main/README.md), [INSTALL.md](https: github.com/casparvitch/dukit/blob/main/INSTALL.md) and [DEVDOCS.md](https: github.com/casparvitch/dukit/blob/main/DEVDOCS.md), or click through the sub-package links below. See the [examples](https: github.com/casparvitch/dukit/tree/main/examples) folder for nice examples of usage. [Repository here](https: github.com/casparvitch/dukit)  DUKIT Public API  Systems ( dukit.systems ) Objects that define the hardware and software configuration of a microscope, and hold the system-specific methods for reading data from the disk. The  System class is abstract, and the other classes inherit from it. The  System class is not intended to be instantiated directly, but rather to be subclassed. If you are from another group, the only thing you should need to do (in the whole package) is write your own System sub-class. -  dukit.systems.System -  dukit.systems.MelbSystem -  dukit.systems.LVControl -  dukit.systems.PyControl -  dukit.systems.Zyla -  dukit.systems.CryoWidefield -  dukit.systems.LegacyCryoWidefield -  dukit.systems.Argus -  dukit.systems.LegacyArgus -  dukit.systems.PyCryoWidefield  System methods -  dukit.systems.System.read_image -  dukit.systems.System.get_hardware_binning -  dukit.systems.System.read_sweep_arr -  dukit.systems.System.get_raw_pixel_size -  dukit.systems.System.get_bias_field -  dukit.systems.System.norm  Working with images ( dukit.itool ) -  dukit.itool.mpl_set_run_config - Set matplotlib rcParams to standardise style of plots. -  dukit.itool.crop_sweep - Crop in frequency/tau direction. -  dukit.itool.crop_roi - Crop to region of interest. -  dukit.itool.smooth_image_stack - Smooth image stack in-frame, doesn't change number of pixels. Can be asymmetric. -  dukit.itool.rebin_image_stack - Rebin image stack in-frame, does change number of pixels. Can be asymmetric. -  dukit.itool.sum_spatially - Sum an image spatially -  dukit.itool.get_im_filtered - Filter an image - currently only gaussian filtering is implemented. -  dukit.itool.get_background - Get background of an image - many different kinds, see function docs. -  dukit.itool.get_colormap_range - Calculate a colormap range for an image. Many kinds, see function docs. -  dukit.itool.plot_image - Base image plotter. -  dukit.itool.plot_image_on_ax - As above, but onto axis already created. -  dukit.itool.mask_polygons - Mask region of image by polygons (see  dukit.polygon module) -  dukit.itool.mu_sigma_inside_polygons - Calculate mean and standard deviation inside polygons. -  dukit.itool.get_aois - Get areas of interest from AOI coords tuple(s).  PL fitting ( dukit.pl ) -  dukit.pl.model.FitModel - Base class for PL fitting models. Subclass this to make your own. -  dukit.pl.model.ConstStretchedExp - T1 model. -  dukit.pl.model.ConstDampedRabi - Rabi model, constant + damped oscillation. -  dukit.pl.model.LinearLorentzians - Linear combination of lorentzians, in init pass number of lorentzians. -  dukit.pl.model.ConstLorentzians - As above, but with constant background. -  dukit.pl.interface.fit_all_pixels - Fit all pixels in an image stack. -  dukit.pl.interface.fit_roi - Fit ROI average spectrum. -  dukit.pl.interface.fit_aois - Fit AOI average spectra. -  dukit.pl.interface.load_fit_results - Load fit results from file.  Field calculations ( dukit.field ) Currently only calculating local field(s) (not vector field) -  dukit.field.defects.Defect - Base class for defects. -  dukit.field.defects.SpinOne - Spin-1 defects general class. -  dukit.field.defects.NVEnsemble - NV ensemble. -  dukit.field.defects.VBEnsemble - VB ensemble. -  dukit.field.defects.SpinPair - Spin pair general class. -  dukit.field.defects.CPairEnsemble - C_? spin pair ensemble. I.e. visible emitter.  Defect methods: -  dukit.field.defects.Defect.b_defects -  dukit.field.defects.Defect.dshift_defects  Drift correction ( dukit.driftcorrect ) -  dukit.driftcorrect.drift_correct_test - Test drift correction params. -  dukit.driftcorrect.drift_correct_measurement - Apply drift correction to image_stack & save output binary.  Magnetic sample simulation ( dukit.magsim ) -  dukit.magsim.SandboxMagSim - Play in an empty sandbox. -  dukit.magsim.ComparisonMagSim - For comparison with a real magnetic field image, to match sim  exp.  Plotting ( dukit.plot ) Need to be imported via  dukit.plot.  to work properly. -  dukit.plot.roi_pl_image -  dukit.plot.aoi_pl_image -  dukit.plot.roi_avg_fits -  dukit.plot.aoi_spectra -  dukit.plot.aoi_spectra_fit -  dukit.plot.pl_param_image -  dukit.plot.pl_param_images  Some matplotlib interactive widgets ( dukit.widget ) -  dukit.widget.LineSelector - Use via  dukit.widget.LinecutSelectionWidget below. -  dukit.widget.PolygonSelector - Use via  dukit.polygon.PolygonSelectionWidget below. -  dukit.widget.LinecutSelectionWidget - Simple linecut selection tool. -  dukit.widget.BulkLinecutWidget - Linecut selection tool for multiple images, but in same slice.  Polygon regions of image, very useful! ( dukit.polygon ) -  dukit.polygon.Polygon - Polygon object, with various useful methods. -  dukit.polygon.PolygonSelectionWidget - Widget allowing you to select polygons on an image. -  dukit.polygon.polygon_selector - GUI interface. -  dukit.polygon.load_polygon_nodes - Load polygon nodes from file.  Fourier tooling ( dukit.fourier ) Currently unused I think, will be useful for vector mag, source recon etc. so leaving in place. -  dukit.fourier.define_k_vectors - Allows asymmetric shape, but haven't tested that yet. -  dukit.fourier.hanning_filter_kspace - Smooth fourier operations below resolvable dimensions, to not amplify noise. -  dukit.fourier.define_current_transform -  dukit.fourier.define_magnetization_transformation -  dukit.fourier.pad_image -  dukit.fourier.unpad_image -  dukit.fourier.MAG_UNIT_CONV - Convert between units of magnetization. -  dukit.fourier.MU_0 - Vacuum permeability.  Other ( dukit.share ) -  dukit.share.RoiAvgFit - Result from ROI average fit(s). -  dukit.share.AoiAvgFit - Result from AOI average fit(s)."
},
{
"ref":"dukit.field",
"url":1,
"doc":"Calculating the magnetic field, from ODMR data. NB: requires fit model to have 'pos' named parameter(s). Have not implemented vector reconstruction etc. yet. Classes    - -  dukit.field.defects.Defect -  dukit.field.defects.SpinOne -  dukit.field.defects.NVEnsemble -  dukit.field.defects.VBEnsemble -  dukit.field.defects.SpinPair -  dukit.field.defects.CPairEnsemble "
},
{
"ref":"dukit.field.defects",
"url":2,
"doc":"This module holds Defect objects, which are used to represent spin defects and their properties, and for extracting 'useful' info from ODMR data. Classes    - -  dukit.field.defects.Defect -  dukit.field.defects.SpinOne -  dukit.field.defects.NVEnsemble -  dukit.field.defects.VBEnsemble -  dukit.field.defects.SpinPair -  dukit.field.defects.CPairEnsemble "
},
{
"ref":"dukit.field.defects.Defect",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.Defect.b_defects",
"url":2,
"doc":"",
"func":1
},
{
"ref":"dukit.field.defects.Defect.dshift_defects",
"url":2,
"doc":"",
"func":1
},
{
"ref":"dukit.field.defects.SpinOne",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.SpinOne.temp_coeff",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.SpinOne.zero_field_splitting",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.SpinOne.gslac",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.SpinOne.gamma",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.SpinOne.b_defects",
"url":2,
"doc":"Calculate magnetic field(s) [in defect frame(s)] in Tesla from resonance frequencies in MHz.",
"func":1
},
{
"ref":"dukit.field.defects.SpinOne.dshift_defects",
"url":2,
"doc":"Calculate d-shifts in MHz from resonance frequencies in MHz.",
"func":1
},
{
"ref":"dukit.field.defects.SpinOne.d_to_T",
"url":2,
"doc":"",
"func":1
},
{
"ref":"dukit.field.defects.NVEnsemble",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.NVEnsemble.temp_coeff",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.NVEnsemble.zero_field_splitting",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.NVEnsemble.gslac",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.NVEnsemble.gamma",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.NVEnsemble.b_defects",
"url":2,
"doc":"Calculate magnetic field(s) [in defect frame(s)] in Tesla from resonance frequencies in MHz.",
"func":1
},
{
"ref":"dukit.field.defects.NVEnsemble.dshift_defects",
"url":2,
"doc":"Calculate d-shifts in MHz from resonance frequencies in MHz.",
"func":1
},
{
"ref":"dukit.field.defects.VBEnsemble",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.VBEnsemble.temp_coeff",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.VBEnsemble.zero_field_splitting",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.VBEnsemble.gslac",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.VBEnsemble.gamma",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.VBEnsemble.b_defects",
"url":2,
"doc":"Calculate magnetic field(s) [in defect frame(s)] in Tesla from resonance frequencies in MHz.",
"func":1
},
{
"ref":"dukit.field.defects.VBEnsemble.dshift_defects",
"url":2,
"doc":"Calculate d-shifts in MHz from resonance frequencies in MHz.",
"func":1
},
{
"ref":"dukit.field.defects.SpinPair",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.SpinPair.gamma",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.SpinPair.b_defects",
"url":2,
"doc":"",
"func":1
},
{
"ref":"dukit.field.defects.CPairEnsemble",
"url":2,
"doc":""
},
{
"ref":"dukit.field.defects.CPairEnsemble.gamma",
"url":2,
"doc":""
},
{
"ref":"dukit.pl",
"url":3,
"doc":"Module for fitting PL data. Cpufit/Gpufit are imported if available. Classes    - -  dukit.pl.model.FitModel -  dukit.pl.model.ConstStretchedExp -  dukit.pl.model.ConstDampedRabi -  dukit.pl.model.LinearLorentzians -  dukit.pl.model.ConstLorentzians Functions     - -  dukit.pl.interface.fit_all_pixels -  dukit.pl.interface.fit_roi -  dukit.pl.interface.fit_aois -  dukit.pl.interface.load_fit_results "
},
{
"ref":"dukit.pl.common",
"url":4,
"doc":"This module holds some functions and classes that are shared between different fitting backends, but are not a part of the user-facing interface. Functions     - -  dukit.pl.common.gen_init_guesses -  dukit.pl.common.bounds_from_range -  dukit.pl.common.calc_sigmas "
},
{
"ref":"dukit.pl.common.gen_init_guesses",
"url":4,
"doc":"Generate initial guesses (and bounds) in fit parameters from options dictionary. Both are returned as dictionaries, you need to use 'gen_{scipy/gpufit/ .}_init_guesses' to convert to the correct (array) format for each specific fitting backend. Arguments     - fit_model : dukit.pl.model.FitModel Fit model you've defined already. guesses_dict : dict Format: key -> list of guesses for each independent version of that fn_type. e.g. 'pos': [ ,  ] for each pos fn_type. bounds_dict : dict Format: key -> bounds for that param type (or use _range). e.g. 'pos_bounds': [5., 25.] or 'pos_range': 5.0 Returns    - init_guesses : dict Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent version of that fn_type. init_bounds : dict Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent version of that fn_type.",
"func":1
},
{
"ref":"dukit.pl.common.bounds_from_range",
"url":4,
"doc":"Generate parameter bounds when given a range option. Arguments     - rang : float or npt.ArrayLike Range for each parameter with name param_key e.g. 'pos_0, pos_1', OR a single value, so each parameter has same range. guess : float/int or array Guess for param, or list of guesses for a given parameter (as for rang) Returns    - bounds : list of lists bounds for each parameter. Dimension depends on dimension of param guess.",
"func":1
},
{
"ref":"dukit.pl.common.calc_sigmas",
"url":4,
"doc":"Calculate fit errors (std. dev.) from jacobian. Arguments     - fit_model : dukit.pl.model.FitModel Fit model you've defined already. sweep_arr : npt.NDArray, 1D Array of sweep values. pl_vec : npt.NDArray, 1D Array of measured PL values. best_params : npt.NDArray, 1D Array of best-fit parameters. Returns    - sigmas : npt.NDArray, 1D Array of standard deviations for each parameter.",
"func":1
},
{
"ref":"dukit.pl.interface",
"url":5,
"doc":"This module holds the general interface tools for fitting data, independent of fit backend (e.g. scipy/gpufit etc.). All of these functions are automatically loaded into the namespace when the fit sub-package is imported. (e.g. import dukit.fit). Functions     - -  dukit.pl.interface.fit_roi -  dukit.pl.interface.fit_aois -  dukit.pl.interface.fit_all_pixels -  dukit.pl.interface.load_fit_results "
},
{
"ref":"dukit.pl.interface.fit_roi",
"url":5,
"doc":"Fit full ROI (region of interest) with the given fit model, for each fit backend that is installed. Arguments     - sig : np array, 3D Sig measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already. ref : np array, 3D Ref measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already. sweep_arr : np array, 1D Affine parameter list (e.g. tau or freq) fit_model : dukit.pl.model.FitModel The model we're fitting to. guess_dict : dict dict holding guesses for each parameter type, e.g. {'pos': [ ,  ], 'amp': [ ,  ],  .} bounds_dict : dict dict holding bound options for each parameter type, e.g. {\"pos_range\": 5.0, \"amp_bounds\": [0.0, 1.0],  .} norm : str default=\"div\" Normalisation method opath : str Output path for saving fit results (.json) Optional parameters passed to scipy least_squares                         - sf_method: str = \"trf\" Fit method, \"trf\", \"dogbox\" or \"lm\" sf_verbose=0 Verbosity of fit -> probably want to keep at 0 sf_gtol: float = 1e-12 Tolerance for termination by the change of the independent variables. sf_xtol: float = 1e-12 Tolerance for termination by the norm of the gradient. sf_ftol: float = 1e-12 Tolerance for termination by the change of the cost function. sf_loss: str = \"linear\" Determines the loss function. This in non trivial check the scipy documentation. Best you don't change this. sf_jac: str | None = None Jacobian. If None uses algebraic jac, otherwise: '2-point' is probably best, then '3-point', 'cs' are other options. Optional parameters passed to cpufit/gpufit                      - gf_tolerance : float = 1e-12 Fit tolerance threshold gf_max_iterations : int = 50 Maximum fit iterations permitted. gf_estimator_id : str = \"LSE\" Estimator to use, \"LSE\" or \"MLE\" (least squares or maximum likelihood). MLE for Poisson, assuming all noise in data is purely Poissonian. Returns    - result : dict Dict: fit_backend str => dukit.share.RoiAvgResult",
"func":1
},
{
"ref":"dukit.pl.interface.fit_aois",
"url":5,
"doc":"Fit AOIs and single pixel with all available fit backends. Arguments     - sig : np array, 3D Sig measurement array, shape: [y, x, sweep_arr]. ref : np array, 3D Ref measurement array, shape: [y, x, sweep_arr]. sweep_arr : np array, 1D Affine parameter list (e.g. tau or freq) fit_model : dukit.pl.model.FitModel The model we're fitting to. guess_dict : dict dict holding guesses for each parameter type, e.g. {'pos': [ ,  ], 'amp': [ ,  ],  .} bounds_dict : dict dict holding bound options for each parameter type, e.g. {\"pos_range\": 5.0, \"amp_bounds\": [0.0, 1.0],  .} aoi_coords : tuple of 4 ints Coordinates of the AOI to fit, in the form (x0, y0, x1, y1). norm : str default=\"div\" Normalisation method opath : str Output path for saving fit results (.json) Optional parameters passed to scipy least_squares                         - sf_method: str = \"trf\" Fit method, \"trf\", \"dogbox\" or \"lm\" sf_verbose=0 Verbosity of fit -> probably want to keep at 0 sf_gtol: float = 1e-12 Tolerance for termination by the change of the independent variables. sf_xtol: float = 1e-12 Tolerance for termination by the norm of the gradient. sf_ftol: float = 1e-12 Tolerance for termination by the change of the cost function. sf_loss: str = \"linear\" Determines the loss function. This in non trivial check the scipy documentation. Best you don't change this. sf_jac: str | None = None Jacobian. If None uses algebraic jac, otherwise: '2-point' is probably best, then '3-point', 'cs' are other options. Optional parameters passed to cpufit/gpufit                      - gf_tolerance : float = 1e-12 Fit tolerance threshold gf_max_iterations : int = 50 Maximum fit iterations permitted. gf_estimator_id : str = \"LSE\" Estimator to use, \"LSE\" or \"MLE\" (least squares or maximum likelihood). MLE for Poisson, assuming all noise in data is purely Poissonian. Returns    - result : dict Dict: aoi_n str => fit_backend => dukit.share.AoiAvgResult",
"func":1
},
{
"ref":"dukit.pl.interface.fit_all_pixels",
"url":5,
"doc":"Fit all pixels in a given image with the given fit model, for each fit backend that is installed. Arguments     - fit_backend : str Name of the fit backend to use, e.g. \"scipyfit\", \"cpufit\", \"gpufit\". sig_norm : np array, 3D Sig measurement array, shape: [y, x, sweep_arr]. sweep_arr : np array, 1D Affine parameter list (e.g. tau or freq) fit_model : dukit.pl.model.FitModel The model we're fitting to. guess_dict : dict dict holding guesses for each parameter type, e.g. {'pos': [ ,  ], 'amp': [ ,  ],  .} bounds_dict : dict dict holding bound options for each parameter type, e.g. {\"pos_range\": 5.0, \"amp_bounds\": [0.0, 1.0],  .} roi_avg_result : dukit.share.RoiAvgFit | None The ROI average fit result, used to get the initial guess for each pixel. If none, uses pguesses as given. odir : str Output  directory (created if doesn't exist) to put .txt files for params. Optional parameters passed to scipy least_squares                         - sf_n_jobs : int = -2 Number of jobs to run in parallel. -1 means use all processors, -2 means all but one, etc. sf_joblib_verbosity : int = 5 Verbosity of joblib parallelisation. sf_method: str = \"trf\" Fit method, \"trf\", \"dogbox\" or \"lm\" sf_verbose : int = 0 Verbosity of fit -> probably want to keep at 0 sf_gtol: float = 1e-12 Tolerance for termination by the change of the independent variables. sf_xtol: float = 1e-12 Tolerance for termination by the norm of the gradient. sf_ftol: float = 1e-12 Tolerance for termination by the change of the cost function. sf_loss: str = \"linear\" Determines the loss function. This in non trivial check the scipy documentation. Best you don't change this. sf_jac: str | None = None Jacobian. If None uses algebraic jac, otherwise: '2-point' is probably best, then '3-point', 'cs' are other options. Optional parameters passed to cpufit/gpufit                      - gf_tolerance : float = 1e-12 Fit tolerance threshold gf_max_iterations : int = 50 Maximum fit iterations permitted. gf_estimator_id : str = \"LSE\" Estimator to use, \"LSE\" or \"MLE\" (least squares or maximum likelihood). MLE for Poisson, assuming all noise in data is purely Poissonian. Returns    - fit_image_results : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. Also has 'residual_0' as a key. Sigmas (stdev on fit error) are given as e.g. pos_0_sigma",
"func":1
},
{
"ref":"dukit.pl.interface.load_fit_results",
"url":5,
"doc":"Load fit results from a json file. Arguments     - idir : str Path to directory containing .txt files. fit_model : dukit.pl.model.FitModel The model we're fitting to. Returns    - fit_results : dict Dict of fit results.",
"func":1
},
{
"ref":"dukit.pl.model",
"url":6,
"doc":"Faster numba-compiled fit model(s). It's a little messy to keep a relatively similar API to old FitModel. TODO include Ella's newer T1 models. Classes    - -  qdmpy.pl.model.FitModel -  qdmpy.pl.model.ConstStretchedExp -  qdmpy.pl.model.ConstDampedRabi -  qdmpy.pl.model.LinearLorentzians -  qdmpy.pl.model.ConstLorentzians "
},
{
"ref":"dukit.pl.model.FitModel",
"url":6,
"doc":"FitModel used to fit to data."
},
{
"ref":"dukit.pl.model.FitModel.residuals_scipyfit",
"url":6,
"doc":"Evaluates residual: fit model (affine params/sweep_arr) - pl values",
"func":1
},
{
"ref":"dukit.pl.model.FitModel.jacobian_scipyfit",
"url":6,
"doc":"Evaluates (analytic) jacobian of fitmodel in format expected by scipy least_squares",
"func":1
},
{
"ref":"dukit.pl.model.FitModel.get_param_defn",
"url":6,
"doc":"Returns list of parameters in fit_model, note there will be duplicates, and they do not have numbers e.g. 'pos_0'. Use  qdmpy.pl.model.FitModel.get_param_odict for that purpose. Returns    - param_defn_ar : tuple List of parameter names (param_defn) in fit model.",
"func":1
},
{
"ref":"dukit.pl.model.FitModel.get_param_odict",
"url":6,
"doc":"get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model Returns    - param_dict : dict Dictionary containing key: params, values: units.",
"func":1
},
{
"ref":"dukit.pl.model.FitModel.get_param_unit",
"url":6,
"doc":"Get unit for a given param_key (given by param_name + \"_\" + param_number) Arguments     - param_name : str Name of parameter, e.g. 'pos' or 'sigma_pos' param_number : float or int Which parameter to use, e.g. 0 for 'pos_0' Returns    - unit : str Unit for that parameter, e.g. \"constant\" -> \"Amplitude (a.u.) ",
"func":1
},
{
"ref":"dukit.pl.model.ConstStretchedExp",
"url":6,
"doc":"FitModel used to fit to data."
},
{
"ref":"dukit.pl.model.ConstStretchedExp.get_param_defn",
"url":6,
"doc":"Returns list of parameters in fit_model, note there will be duplicates, and they do not have numbers e.g. 'pos_0'. Use  qdmpy.pl.model.FitModel.get_param_odict for that purpose. Returns    - param_defn_ar : tuple List of parameter names (param_defn) in fit model.",
"func":1
},
{
"ref":"dukit.pl.model.ConstStretchedExp.get_param_odict",
"url":6,
"doc":"get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model Returns    - param_dict : dict Dictionary containing key: params, values: units.",
"func":1
},
{
"ref":"dukit.pl.model.ConstStretchedExp.residuals_scipyfit",
"url":6,
"doc":"Evaluates residual: fit model (affine params/sweep_arr) - pl values",
"func":1
},
{
"ref":"dukit.pl.model.ConstStretchedExp.jacobian_scipyfit",
"url":6,
"doc":"Evaluates (analytic) jacobian of fitmodel in format expected by scipy least_squares",
"func":1
},
{
"ref":"dukit.pl.model.ConstStretchedExp.get_param_unit",
"url":6,
"doc":"Get unit for a given param_key (given by param_name + \"_\" + param_number) Arguments     - param_name : str Name of parameter, e.g. 'pos' or 'sigma_pos' param_number : float or int Which parameter to use, e.g. 0 for 'pos_0' Returns    - unit : str Unit for that parameter, e.g. \"constant\" -> \"Amplitude (a.u.) ",
"func":1
},
{
"ref":"dukit.pl.model.ConstDampedRabi",
"url":6,
"doc":"FitModel used to fit to data."
},
{
"ref":"dukit.pl.model.ConstDampedRabi.get_param_defn",
"url":6,
"doc":"Returns list of parameters in fit_model, note there will be duplicates, and they do not have numbers e.g. 'pos_0'. Use  qdmpy.pl.model.FitModel.get_param_odict for that purpose. Returns    - param_defn_ar : tuple List of parameter names (param_defn) in fit model.",
"func":1
},
{
"ref":"dukit.pl.model.ConstDampedRabi.get_param_odict",
"url":6,
"doc":"get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model Returns    - param_dict : dict Dictionary containing key: params, values: units.",
"func":1
},
{
"ref":"dukit.pl.model.ConstDampedRabi.residuals_scipyfit",
"url":6,
"doc":"Evaluates residual: fit model (affine params/sweep_arr) - pl values",
"func":1
},
{
"ref":"dukit.pl.model.ConstDampedRabi.jacobian_scipyfit",
"url":6,
"doc":"Evaluates (analytic) jacobian of fitmodel in format expected by scipy least_squares",
"func":1
},
{
"ref":"dukit.pl.model.ConstDampedRabi.get_param_unit",
"url":6,
"doc":"Get unit for a given param_key (given by param_name + \"_\" + param_number) Arguments     - param_name : str Name of parameter, e.g. 'pos' or 'sigma_pos' param_number : float or int Which parameter to use, e.g. 0 for 'pos_0' Returns    - unit : str Unit for that parameter, e.g. \"constant\" -> \"Amplitude (a.u.) ",
"func":1
},
{
"ref":"dukit.pl.model.LinearLorentzians",
"url":6,
"doc":"FitModel used to fit to data."
},
{
"ref":"dukit.pl.model.LinearLorentzians.residuals_scipyfit",
"url":6,
"doc":"Evaluates residual: fit model (affine params/sweep_arr) - pl values",
"func":1
},
{
"ref":"dukit.pl.model.LinearLorentzians.jacobian_scipyfit",
"url":6,
"doc":"Evaluates (analytic) jacobian of fitmodel in format expected by scipy least_squares",
"func":1
},
{
"ref":"dukit.pl.model.LinearLorentzians.get_param_defn",
"url":6,
"doc":"Returns list of parameters in fit_model, note there will be duplicates, and they do not have numbers e.g. 'pos_0'. Use  qdmpy.pl.model.FitModel.get_param_odict for that purpose. Returns    - param_defn_ar : tuple List of parameter names (param_defn) in fit model.",
"func":1
},
{
"ref":"dukit.pl.model.LinearLorentzians.get_param_odict",
"url":6,
"doc":"get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model Returns    - param_dict : dict Dictionary containing key: params, values: units.",
"func":1
},
{
"ref":"dukit.pl.model.LinearLorentzians.get_param_unit",
"url":6,
"doc":"Get unit for a given param_key (given by param_name + \"_\" + param_number) Arguments     - param_name : str Name of parameter, e.g. 'pos' or 'sigma_pos' param_number : float or int Which parameter to use, e.g. 0 for 'pos_0' Returns    - unit : str Unit for that parameter, e.g. \"constant\" -> \"Amplitude (a.u.) ",
"func":1
},
{
"ref":"dukit.pl.model.ConstLorentzians",
"url":6,
"doc":"FitModel used to fit to data."
},
{
"ref":"dukit.pl.model.ConstLorentzians.residuals_scipyfit",
"url":6,
"doc":"Evaluates residual: fit model (affine params/sweep_arr) - pl values",
"func":1
},
{
"ref":"dukit.pl.model.ConstLorentzians.jacobian_scipyfit",
"url":6,
"doc":"Evaluates (analytic) jacobian of fitmodel in format expected by scipy least_squares",
"func":1
},
{
"ref":"dukit.pl.model.ConstLorentzians.get_param_defn",
"url":6,
"doc":"Returns list of parameters in fit_model, note there will be duplicates, and they do not have numbers e.g. 'pos_0'. Use  qdmpy.pl.model.FitModel.get_param_odict for that purpose. Returns    - param_defn_ar : tuple List of parameter names (param_defn) in fit model.",
"func":1
},
{
"ref":"dukit.pl.model.ConstLorentzians.get_param_odict",
"url":6,
"doc":"get ordered dict of key: param_key (param_name), val: param_unit for all parameters in fit_model Returns    - param_dict : dict Dictionary containing key: params, values: units.",
"func":1
},
{
"ref":"dukit.pl.model.ConstLorentzians.get_param_unit",
"url":6,
"doc":"Get unit for a given param_key (given by param_name + \"_\" + param_number) Arguments     - param_name : str Name of parameter, e.g. 'pos' or 'sigma_pos' param_number : float or int Which parameter to use, e.g. 0 for 'pos_0' Returns    - unit : str Unit for that parameter, e.g. \"constant\" -> \"Amplitude (a.u.) ",
"func":1
},
{
"ref":"dukit.pl.scipyfit",
"url":7,
"doc":"This module holds tools for fitting raw data via scipy. (scipy backend) Functions     - -  dukit.pl.scipyfit.fit_roi_avg_pl -  dukit.pl.scipyfit.fit_aois_pl -  qdmpy.pl.scipyfit.fit_all_pixels_pl -  dukit.pl.scipyfit._gen_sf_guesses_bounds -  dukit.pl.scipyfit._spfitter "
},
{
"ref":"dukit.pl.scipyfit.fit_roi_avg_pl",
"url":7,
"doc":"Fit AOI averages Arguments     - sig : np array, 3D Sig measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already. ref : np array, 3D Ref measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already. sweep_arr : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  dukit.pl.model.FitModel The model we're fitting to. guess_dict : dict dict holding guesses for each parameter type, e.g. {'pos': [ ,  ], 'amp': [ ,  ],  .} bounds_dict : dict dict holding bound options for each parameter type, e.g. {\"pos_range\": 5.0, \"amp_bounds\": [0.0, 1.0],  .} norm : str default=\"div\" Normalisation method Optional parameters passed to scipy least_squares                         - method: str = \"trf\" Fit method, \"trf\", \"dogbox\" or \"lm\" verbose: int =0 Verbosity of fit -> probably want to keep at 0 gtol: float = 1e-12 Tolerance for termination by the change of the independent variables. xtol: float = 1e-12 Tolerance for termination by the norm of the gradient. ftol: float = 1e-12 Tolerance for termination by the change of the cost function. loss: str = \"linear\" Determines the loss function. This in non trivial check the scipy documentation. Best you don't change this. jac: str | None = None Jacobian. If None uses algebraic jac, otherwise: '2-point' is probably best, then '3-point', 'cs' are other options. Returns    - fit_image_results : dukit.share.RoiAvgFit",
"func":1
},
{
"ref":"dukit.pl.scipyfit.fit_aois_pl",
"url":7,
"doc":"Fit AOI averages Arguments     - sig : np array, 3D Sig measurement array, shape: [y, x, sweep_arr]. ref : np array, 3D Ref measurement array, shape: [y, x, sweep_arr]. sweep_arr : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  dukit.pl.model.FitModel The model we're fitting to. guess_dict : dict dict holding guesses for each parameter type, e.g. {'pos': [ ,  ], 'amp': [ ,  ],  .} bounds_dict : dict dict holding bound options for each parameter type, e.g. {\"pos_range\": 5.0, \"amp_bounds\": [0.0, 1.0],  .}  aoi_coords : tuple of 4-tuples As elsewhere norm : str default=\"div\" Normalisation method Optional parameters passed to scipy least_squares                         - method: str = \"trf\" Fit method, \"trf\", \"dogbox\" or \"lm\" verbose=0 Verbosity of fit -> probably want to keep at 0 gtol: float = 1e-12 Tolerance for termination by the change of the independent variables. xtol: float = 1e-12 Tolerance for termination by the norm of the gradient. ftol: float = 1e-12 Tolerance for termination by the change of the cost function. loss: str = \"linear\" Determines the loss function. This in non trivial check the scipy documentation. Best you don't change this. jac: str | None = None Jacobian. If None uses algebraic jac, otherwise: '2-point' is probably best, then '3-point', 'cs' are other options. Returns    - fit_image_results : dict Format: {\"AOI_n\": {\"scipyfit\": AoiAvgFit},  .}",
"func":1
},
{
"ref":"dukit.pl.scipyfit.fit_all_pixels_pl",
"url":7,
"doc":"Fits each pixel and returns dictionary of param_name -> param_image. Arguments     - sig_norm : np array, 3D Normalised measurement array, shape: [sweep_arr, y, x]. sweep_arr : np array, 1D Affine parameter list (e.g. tau or freq) fit_model :  dukit.pl.model.FitModel The model we're fitting to. guess_dict : dict dict holding guesses for each parameter type, e.g. {'pos': [ ,  ], 'amp': [ ,  ],  .} bounds_dict : dict dict holding bound options for each parameter type, e.g. {\"pos_range\": 5.0, \"amp_bounds\": [0.0, 1.0],  .} roi_avg_result : dukit.share.RoiAvgFit | None The result of fitting the ROI average. If done, directly uses guesses provided. n_jobs : int, default=-2 Number of jobs to run concurrently, see joblib docs. -2  = leaving one cpu free, etc. for neg numbers. joblib_verbosity:int = 5 How often to update progress bar. Optional parameters passed to scipy least_squares                         - method: str = \"trf\" Fit method, \"trf\", \"dogbox\" or \"lm\" verbose=0 Verbosity of fit -> probably want to keep at 0 gtol: float = 1e-12 Tolerance for termination by the change of the independent variables. xtol: float = 1e-12 Tolerance for termination by the norm of the gradient. ftol: float = 1e-12 Tolerance for termination by the change of the cost function. loss: float = \"linear\" Determines the loss function. This in non trivial check the scipy documentation. Best you don't change this. jac: str | None = None Jacobian. If None uses algebraic jac, otherwise: '2-point' is probably best, then '3-point', 'cs' are other options. Returns    - fit_image_results : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. Also has 'residual_0' as a key. Sigmas (stdev on fit error) are given as e.g. pos_0_sigma",
"func":1
},
{
"ref":"dukit.pl.scipyfit._spfitter",
"url":7,
"doc":"Single pixel fitter. ~ Fit inputs -> (params; param_std_dev; sum(resid @ solution.",
"func":1
},
{
"ref":"dukit.pl.scipyfit._gen_sf_guesses_bounds",
"url":7,
"doc":"Generate arrays of initial fit guesses and bounds in correct form for scipy least_squares. init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays, that scipy will recognise. In particular, we specificy that each of the 'num' of each 'fn_type' have independent parameters, so must have independent init_guesses and init_bounds when plugged into scipy. Arguments     - fit_model : dukit.pl.model.FitModel Model definition. init_guesses : dict Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent version of that fn_type. init_bounds : dict Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent version of that fn_type. Returns    - fit_param_ar : np array, shape: num_params The initial fit parameter guesses. fit_param_bound_ar : np array, shape: (num_params, 2) Fit parameter bounds.",
"func":1
},
{
"ref":"dukit.pl.cpufit",
"url":8,
"doc":"This module holds tools for fitting raw data via scipy. (scipy backend) Functions     - -  dukit.pl.cpufit.fit_roi_avg_pl -  dukit.pl.cpufit.fit_aois_pl -  qdmpy.pl.cpufit.fit_all_pixels_pl -  dukit.pl.cpufit._gen_cf_guesses_bounds "
},
{
"ref":"dukit.pl.cpufit.fit_roi_avg_pl",
"url":8,
"doc":"Fit AOI averages Arguments     - sig : np array, 3D Sig measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already. ref : np array, 3D Ref measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already. sweep_arr : np array, 1D Affine parameter list (e.g. tau or freq) fit_model : dukit.pl.model.FitModel The model we're fitting to. guess_dict : dict dict holding guesses for each parameter type, e.g. {'pos': [ ,  ], 'amp': [ ,  ],  .} bounds_dict : dict dict holding bound options for each parameter type, e.g. {\"pos_range\": 5.0, \"amp_bounds\": [0.0, 1.0],  .} norm : str default=\"div\" Normalisation method Optional parameters passed to cpufit                   tolerance : float = 1e-12 Fit tolerance threshold max_iterations : int = 50 Maximum fit iterations permitted. estimator_id : str = \"LSE\" Estimator to use, \"LSE\" or \"MLE\" (least squares or maximum likelihood). MLE for Poisson, assuming all noise in data is purely Poissonian. Returns    - fit_image_results : dukit.share.RoiAvgFit",
"func":1
},
{
"ref":"dukit.pl.cpufit.fit_aois_pl",
"url":8,
"doc":"Fit AOI averages Arguments     - sig : np array, 3D Sig measurement array, shape: [y, x, sweep_arr]. ref : np array, 3D Ref measurement array, shape: [y, x, sweep_arr]. sweep_arr : np array, 1D Affine parameter list (e.g. tau or freq) fit_model : dukit.pl.model.FitModel The model we're fitting to. guess_dict : dict dict holding guesses for each parameter type, e.g. {'pos': [ ,  ], 'amp': [ ,  ],  .} bounds_dict : dict dict holding bound options for each parameter type, e.g. {\"pos_range\": 5.0, \"amp_bounds\": [0.0, 1.0],  .}  aoi_coords : tuple of 4-tuples As elsewhere norm : str default=\"div\" Normalisation method Optional parameters passed to cpufit                   tolerance : float = 1e-12 Fit tolerance threshold max_iterations : int = 50 Maximum fit iterations permitted. estimator_id : str = \"LSE\" Estimator to use, \"LSE\" or \"MLE\" (least squares or maximum likelihood). MLE for Poisson, assuming all noise in data is purely Poissonian. Returns    - fit_image_results : dict Format: {\"AOI_n\": {\"scipyfit\": AoiAvgFit},  .}",
"func":1
},
{
"ref":"dukit.pl.cpufit.fit_all_pixels_pl",
"url":8,
"doc":"Fits each pixel and returns dictionary of param_name -> param_image. Arguments     - sig_norm : np array, 3D Normalised measurement array, shape: [sweep_arr, y, x]. sweep_arr : np array, 1D Affine parameter list (e.g. tau or freq) fit_model : dukit.pl.model.FitModel The model we're fitting to. guess_dict : dict Format: key -> list of guesses for each independent version of that fn_type. e.g. 'pos': [ ,  ] for each pos fn_type. bounds_dict : dict Format: key -> bounds for that param type (or use _range). e.g. 'pos_bounds': [5., 25.] or 'pos_range': 5.0 roi_avg_result : dukit.share.RoiAvgFit | None The result of fitting the ROI average. If done, directly uses guesses provided. n_jobs : int, default=-2 Number of jobs to run concurrently, see joblib docs. -2  = leaving one cpu free, etc. for neg numbers. joblib_verbosity:int = 5 How often to update progress bar. Optional parameters passed to cpufit                   tolerance : float = 1e-12 Fit tolerance threshold max_iterations : int = 50 Maximum fit iterations permitted. estimator_id : str = \"LSE\" Estimator to use, \"LSE\" or \"MLE\" (least squares or maximum likelihood). MLE for Poisson, assuming all noise in data is purely Poissonian. Returns    - fit_image_results : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. Also has 'residual_0' as a key. Sigmas (stdev on fit error) are given as e.g. pos_0_sigma",
"func":1
},
{
"ref":"dukit.pl.cpufit._gen_cf_guesses_bounds",
"url":8,
"doc":"Generate arrays of initial fit guesses and bounds in correct form for cpufit init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays, that scipy will recognise. In particular, we specificy that each of the 'num' of each 'fn_type' have independent parameters, so must have independent init_guesses and init_bounds when plugged into scipy. Arguments     - fit_model : dukit.pl.model.FitModel Model definition. init_guesses : dict Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent version of that fn_type. init_bounds : dict Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent version of that fn_type. Returns    - fit_param_ar : np array, shape: num_params The initial fit parameter guesses. fit_param_bound_ar : np array, shape: (num_params, 2) Fit parameter bounds.",
"func":1
},
{
"ref":"dukit.driftcorrect",
"url":9,
"doc":"This module holds tools for measurement (lateral) drift correction. Requires a  dukit.systems.System for reading files from disk. Most of the functions are not documented, but the API is only 2 funcs: Functions     - -  dukit.driftcorrect.drift_correct_test -  dukit.driftcorrect.drift_correct_measurement "
},
{
"ref":"dukit.driftcorrect.read_and_drift_correct",
"url":9,
"doc":"",
"func":1
},
{
"ref":"dukit.driftcorrect.drift_correct_measurement",
"url":9,
"doc":"Arguments     - directory : str Path to directory that contains all the measurements. start_num : int Image/measurement to start accumulating from. end_num : int Image/measurement to end accumulation. stub : Callable int], str] Function that takes image num and returns path to that measurement from directory. I.e. directory + stub(X) for filepath ass. with 'X' output_file : str Output will be stored in directory + output file system : dukit.systems.System object Used for reading in files output_file : str Where to save the drift-corrected binary Format is based on 'system'. feature_roi_coords : tuple[int, int, int, int] Define roi, here a feature in PL that is used from cross-correlation. start_x, start_y, end_x, end_y. -1 to use edge of image on that side. Here the ROI is only used as a PL 'feature'/'window' for cross-correlation drift correction, not for cropping output etc. image_nums_mask : 1D ndarray Same shape as list(range(start_num, end_num + 1 . Where false, don't include that image in accumulation.",
"func":1
},
{
"ref":"dukit.driftcorrect.drift_correct_test",
"url":9,
"doc":"Test the drift correction on a subset (comparison_nums) of the measurements. Arguments     - directory : str Path to directory that contains all the measurements. start_num : int Image/measurement to start accumulating from. (Also the reference frame) end_num : int Image/measurement to end accumulation. comparison_nums : list of ints List of image/measurment nums to compare drift calc on. stub : function Function that takes image num and returns path to that measurement from directory. I.e. directory + stub(X) for filepath ass. with 'X' system : dukit.systems.System object Used for reading in files ignore_ref : bool = False feature_roi_coords : tuple[int, int, int, int] Define roi, here a feature in PL that is used from cross-correlation. start_x, start_y, end_x, end_y. -1 to use edge of image on that side. Here the ROI is only used as a PL 'feature'/'window' for cross-correlation drift correction, not for cropping output etc. Returns    - crop_fig : plt.Figure For further editing/saving etc. crop_axs : plt.Axes For further editing/saving etc.",
"func":1
},
{
"ref":"dukit.magsim",
"url":10,
"doc":"Interface to mag simulations. TODO needs documenting/cleaning Basically add some examples HERE of how to use this subpkg. Apologies for lack of type information, I don't understand everything sufficiently and it currently  just works . Classes    - -  dukit.magsim.MagSim -  dukit.magsim.SandboxMagSim -  dukit.magsim.ComparisonMagSim "
},
{
"ref":"dukit.magsim.MagSim",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.base_image",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.polygon_nodes",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.mag",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.template_polygon_nodes",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.bfield",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.standoff",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.magnetizations_lst",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.unit_vectors_lst",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.ny",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.nx",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.pixel_size",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.MagSim.add_polygons",
"url":10,
"doc":"polygons is dict (polygons directly) or str (path to)",
"func":1
},
{
"ref":"dukit.magsim.MagSim.select_polygons",
"url":10,
"doc":"manually select polygons",
"func":1
},
{
"ref":"dukit.magsim.MagSim.save_polygons",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.define_magnets",
"url":10,
"doc":"magnetizations: int/float if the same for all polygons, or an iterable of len(polygon_nodes) -> in units of mu_b / nm^2 (or mu_b / PX^2 for SandboxMagSim) unit_vectors: 3-iterable if the same for all polygons (cartesian coords), or an iterable of len(polygon_nodes) each element a 3-iterable",
"func":1
},
{
"ref":"dukit.magsim.MagSim.save_magnets",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.load_magnets",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.run",
"url":10,
"doc":"Everything units of metres.",
"func":1
},
{
"ref":"dukit.magsim.MagSim.get_bfield_im",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.get_magnetization_im",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.plot_magsim_magnetization",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.plot_magsim_magnetizations",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.plot_magsim_bfield_at_nvs",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.crop_polygons",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.crop_polygons_gui",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.crop_magnetization",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.crop_domains",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.MagSim.crop_magnetization_gui",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.SandboxMagSim",
"url":10,
"doc":"Image conventions: first index is height."
},
{
"ref":"dukit.magsim.SandboxMagSim.add_template_polygons",
"url":10,
"doc":"polygons takes precedence.",
"func":1
},
{
"ref":"dukit.magsim.SandboxMagSim.rescale_template",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.SandboxMagSim.adjust_template",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.SandboxMagSim.set_template_as_polygons",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.SandboxMagSim.add_polygons",
"url":10,
"doc":"polygons is dict (polygons directly) or str (path to)",
"func":1
},
{
"ref":"dukit.magsim.SandboxMagSim.select_polygons",
"url":10,
"doc":"manually select polygons",
"func":1
},
{
"ref":"dukit.magsim.SandboxMagSim.define_magnets",
"url":10,
"doc":"magnetizations: int/float if the same for all polygons, or an iterable of len(polygon_nodes) -> in units of mu_b / nm^2 (or mu_b / PX^2 for SandboxMagSim) unit_vectors: 3-iterable if the same for all polygons (cartesian coords), or an iterable of len(polygon_nodes) each element a 3-iterable",
"func":1
},
{
"ref":"dukit.magsim.SandboxMagSim.run",
"url":10,
"doc":"Everything units of metres.",
"func":1
},
{
"ref":"dukit.magsim.ComparisonMagSim",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.ComparisonMagSim.unscaled_polygon_nodes",
"url":10,
"doc":""
},
{
"ref":"dukit.magsim.ComparisonMagSim.rescale",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.ComparisonMagSim.plot_comparison",
"url":10,
"doc":"",
"func":1
},
{
"ref":"dukit.magsim.ComparisonMagSim.add_polygons",
"url":10,
"doc":"polygons is dict (polygons directly) or str (path to)",
"func":1
},
{
"ref":"dukit.magsim.ComparisonMagSim.select_polygons",
"url":10,
"doc":"manually select polygons",
"func":1
},
{
"ref":"dukit.magsim.ComparisonMagSim.define_magnets",
"url":10,
"doc":"magnetizations: int/float if the same for all polygons, or an iterable of len(polygon_nodes) -> in units of mu_b / nm^2 (or mu_b / PX^2 for SandboxMagSim) unit_vectors: 3-iterable if the same for all polygons (cartesian coords), or an iterable of len(polygon_nodes) each element a 3-iterable",
"func":1
},
{
"ref":"dukit.magsim.ComparisonMagSim.run",
"url":10,
"doc":"Everything units of metres.",
"func":1
},
{
"ref":"dukit.rebin",
"url":11,
"doc":"Python/NumPy implementation of IDL's rebin function. NOTE: vendored from https: github.com/sbrisard/rebin as it has no conda-forge package     Python/NumPy implementation of IDL's rebin function. See http: www.harrisgeospatial.com/docs/rebin.html. The  rebin function defined in this module first groups the cells of the input array in tiles of specified size. Then, a reduction function is applied to each tile, which is replaced by a single value. The resulting array is returned: its dimensions are the number of tiles in the input array. Rebin is released under a BSD 3-clause license. Rationale     - The input array,  a is assumed to be  strided . In other words, if  a.strides = (s0, s1,  .), then  a[i0, i1,  .] = a s0 i0 + s1 i1 +  . , where    .  denotes the offset operator. To compute the output array, we first create a tiled version of  a . The number of dimensions of  tiled is twice that of  a : for each index in  a ,  tiled has one  slow index and one  fast index  tiled[i0, i1,  ., j0, j1,  .] = a[f0 i0 + j0, f1 i1 + j1,  .], where  factor=(f0, f1,  .) is the binning factor (size of the tiles). Upon using the strides of  a  tiled[i0, i1,  ., j0, j1,  .] = a s0 f0 i0 + s1 f1 i1 +  . + s0 j0 + s1 j1 +  . , which shows that the strides of  tiled are  tiled.strides = (s0 f0, s1 f1,  ., s0, s1,  .). In other words,  tiled is a  view of  a with modified strides. Restriding an array can be done with the  as_strided function from  numpy.lib.stride_tricks . Then, the output array is readily computed as follows  out = func(tiled, axis = tuple(range(-a.ndim, 0 ) where reduction is carried out on the fast indices. Boundary cases        When the dimensions of the input array are not integer multiples of the dimensions of the tiles, the remainding rows/columns are simply discarded. For example  +    +    +    +    +  + | 1 1 | 2 2 | 3 3 | 4 4 | 5 | | 1 1 | 2 2 | 3 3 | 4 4 | 5 | +    +    +    +    +  + | 6 6 | 7 7 | 8 8 | 9 9 | 10 | | 6 6 | 7 7 | 8 8 | 9 9 | 10 | +    +    +    +    +  + | 11 11 | 12 12 | 13 13 | 14 14 | 15 | +    +    +    +    +  + will produce  +  +  +  +  + | 4 | 8 | 12 | 16 | +  +  +  +  + | 24 | 28 | 32 | 36 | +  +  +  +  + for (2, 2) tiles and a  sum reduction."
},
{
"ref":"dukit.rebin.rebin",
"url":11,
"doc":"Aggregate data from the input array  a into rectangular tiles. The output array results from tiling  a and applying  func to each tile.  factor specifies the size of the tiles. More precisely, the returned array  out is such that out[i0, i1,  .] = func(a[f0 i0:f0 (i0+1), f1 i1:f1 (i1+1),  .]) If  factor is an integer-like scalar, then  f0 = f1 =  . = factor in the above formula. If  factor is a sequence of integer-like scalars, then  f0 = factor[0] ,  f1 = factor[1] ,  . and the length of  factor must equal the number of dimensions of  a . The reduction function  func must accept an  axis argument. Examples of such function are -  numpy.mean (default), -  numpy.sum , -  numpy.product , -  . The following example shows how a (4, 6) array is reduced to a (2, 2) array >>> import numpy >>> from rebin import rebin >>> a = numpy.arange(24).reshape(4, 6) >>> rebin(a, factor=(2, 3), func=numpy.sum) array( 24, 42], [ 96, 114 ) If the elements of  factor are not integer multiples of the dimensions of  a , the remainding cells are discarded. >>> rebin(a, factor=(2, 2), func=numpy.sum) array( 16, 24, 32], [72, 80, 88 )",
"func":1
},
{
"ref":"dukit.geom",
"url":12,
"doc":"This module holds tools for determining the geometry of the defect-bias field system etc., required for retrieving/reconstructing vector fields. Currently only written for NVs. & Currently not utilized anywhere. NOTE probably shouldn't be exposed as API TODO come back and check this is sufficient for ham fit / dc_odmr etc. Probably want to add these details to  dukit.field.defects.Defect objects. Functions     - -  dukit.geom.get_unvs -  dukit.geom.get_unv_frames Constants     - -  dukit.geom.NV_AXES_100_110 -  dukit.geom.NV_AXES_100_100 -  dukit.geom.NV_AXES_111 "
},
{
"ref":"dukit.geom.NV_AXES_100_110",
"url":12,
"doc":" top face oriented,  edge face oriented diamond (CVD). NV orientations (unit vectors) relative to lab frame. Assuming diamond is square to lab frame: first 3 numbers: orientation of top face of diamond, e.g.  second 3 numbers: orientation of edges of diamond, e.g.  CVD Diamonds are usually  ,  . HPHT usually  ,  . ![](https: i.imgur.com/Rudnzyo.png) Purple plane corresponds to top (or bottom) face of diamond, orange planes correspond to edge faces."
},
{
"ref":"dukit.geom.NV_AXES_100_100",
"url":12,
"doc":" top face oriented,  edge face oriented diamond (HPHT). NV orientations (unit vectors) relative to lab frame. Assuming diamond is square to lab frame: first 3 numbers: orientation of top face of diamond, e.g.  second 3 numbers: orientation of edges of diamond, e.g.  CVD Diamonds are usually  ,  . HPHT usually  ,  . ![](https: i.imgur.com/cpErjAH.png) Purple plane: top face of diamond, orange plane: edge faces."
},
{
"ref":"dukit.geom.NV_AXES_111",
"url":12,
"doc":" top face oriented. NV orientations (unit vectors) relative to lab frame. Only the first nv can be oriented in general. This constant is defined as a convenience for single-bnv  measurements.  diamonds have an NV family oriented in z, i.e. perpindicular to the diamond surface."
},
{
"ref":"dukit.geom.get_unvs",
"url":12,
"doc":"Returns orientation (relative to lab frame) of NVs. Shape: (4,3) regardless of sample. Arguments     - TODO Returns    - unvs : np array Shape: (4,3). Equivalent to uNV_Z for each NV. (Sorted largest to smallest Bnv)",
"func":1
},
{
"ref":"dukit.geom.get_unv_frames",
"url":12,
"doc":"Returns array representing each NV reference frame. I.e. each index is a 2D array: [uNV_X, uNV_Y, uNV_Z] representing the unit vectors for that NV reference frame, in the lab frame. Arguments     - TODO Returns    - unv_frames : np array [ [uNV1_X, uNV1_Y, uNV1_Z], [uNV2_X, uNV2_Y, uNV2_Z],  .]",
"func":1
},
{
"ref":"dukit.fourier",
"url":13,
"doc":"Shared FFTW tooling. Functions     - -  dukit.fourier.unpad_image -  dukit.fourier.pad_image -  dukit.fourier.define_k_vectors -  dukit.fourier.set_naninf_to_zero -  dukit.fourier.hanning_filter_kspace -  dukit.fourier.define_magnetization_transformation -  dukit.fourier.define_current_transform Constants     - -  dukit.fourier.MAG_UNIT_CONV -  dukit.fourier.MU_0 "
},
{
"ref":"dukit.fourier.MAG_UNIT_CONV",
"url":13,
"doc":"Convert unit for magnetization to something more helpful. SI unit measured: Amps: A [for 2D magnetization, A/m for 3D] More useful: Bohr magnetons per nanometre squared: mu_B nm^-2   mu_B -> 9.274 010 e-24 A m^+2 or J/T m^2 -> 1e+18 nm^2 Measure x amps = x A def mu_B = 9.2_ in units of A m^2 => x A = x (1 / 9.2_) in units of mu_B/m^2 => x A = x (1e-18/9.2_) in units of mu_B/nm^2  "
},
{
"ref":"dukit.fourier.MU_0",
"url":13,
"doc":"Vacuum permeability"
},
{
"ref":"dukit.fourier.unpad_image",
"url":13,
"doc":"undo a padding defined by  dukit.fourier.pad_image (it returns the padder list)",
"func":1
},
{
"ref":"dukit.fourier.pad_image",
"url":13,
"doc":"pad_mode -> see np.pad pad_factor -> either side of image",
"func":1
},
{
"ref":"dukit.fourier.define_k_vectors",
"url":13,
"doc":"Get scaled k vectors (as meshgrid) for fft. Arguments      shape : list Shape of fft array to get k vectors for. raw_pixel_size : float I.e. camera pixel size applied_binning : 2-tuple of ints or int Binning that has been applied. 2-tuple for asymmetric binning NOTE sorry untested in fourier currently k_vector_epsilon : float Add an epsilon value to the k-vectors to avoid some issues with 1/0. Returns    - ky, kx, k : np array Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )",
"func":1
},
{
"ref":"dukit.fourier.set_naninf_to_zero",
"url":13,
"doc":"replaces NaNs and infs with zero",
"func":1
},
{
"ref":"dukit.fourier.hanning_filter_kspace",
"url":13,
"doc":"Computes a hanning image filter with both low and high pass filters. Arguments     - k : np array Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 ) do_filt : bool Do a hanning filter? hanning_high_cutoff : float Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff. hanning_low_cutoff : float Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff. standoff : float Distance NV layer  Sample. Returns    - img_filter : (2d array, float) bandpass filter to remove artifacts in the FFT process.",
"func":1
},
{
"ref":"dukit.fourier.define_magnetization_transformation",
"url":13,
"doc":"M => b fourier-space transformation. Parameters      ky, kx, k : np array Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 ) standoff : float Distance NV layer  Sample nv_layer_thickness : float or None, default : None Thickness of NV layer (in metres) Returns    - d_matrix : np array Transformation such that B = d_matrix  m. E.g. for z magnetized sample: m_to_bnv = ( unv[0]  d_matrix[2, 0,  ] + unv[1]  d_matrix[2, 1,  ] + unv[2]  d_matrix[2, 2,  ] ) -> First index '2' is for z magnetization (see m_from_bxy for in-plane mag process), the second index is for the (bnv etc.) bfield axis (0:x, 1:y, 2:z), and the last index iterates through the k values/vectors. See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"dukit.fourier.define_current_transform",
"url":13,
"doc":"b => J fourier-space transformation. Arguments     - u_proj : 3-tuple Shape: 3, the direction the magnetic field was measured in (projected onto). ky, kx, k : np arrays Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 ) standoff : float or None, default : None Distance NV layer  sample nv_layer_thickness : float or None, default : None Thickness of NV layer (in metres) Returns    - b_to_jx, b_to_jy : np arrays (2D) See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020). https: doi.org/10.1103/PhysRevApplied.14.024076 https: arxiv.org/abs/2005.06788",
"func":1
},
{
"ref":"dukit.json2dict",
"url":14,
"doc":"json2dict; functions for loading json files to dicts and the inverse. Functions     - -  dukit.json2dict.json_to_dict -  dukit.json2dict.dict_to_json -  dukit.json2dict.dict_to_json_str -  dukit.json2dict.fail_float "
},
{
"ref":"dukit.json2dict.json_to_dict",
"url":14,
"doc":"read the json file at filepath into a dict",
"func":1
},
{
"ref":"dukit.json2dict.dict_to_json",
"url":14,
"doc":"save the dict as a json in a pretty way",
"func":1
},
{
"ref":"dukit.json2dict.dict_to_json_str",
"url":14,
"doc":"",
"func":1
},
{
"ref":"dukit.json2dict.fail_float",
"url":14,
"doc":"Used in particular for reading the metadata to convert all numbers into floats and leave strings as strings.",
"func":1
},
{
"ref":"dukit.share",
"url":15,
"doc":"Data structures/types shared between other modules. Classes    - -  dukit.share.AoiAvgFit -  dukit.share.RoiAvgFit "
},
{
"ref":"dukit.share.AoiAvgFit",
"url":15,
"doc":"Holds result from an AOI fit. Only a method to save. Attributes      aoi_num : int Number associated with this AOI sweep_arr : ndarray, 1D Freqs/taus avg_sig_norm : ndarray, 1D sig_norm (trace/spectrum) averaged over this aoi avg_sig : ndarray, 1D sig (trace/spectrum) averaged over this aoi avg_ref : ndarray, 1D ref (trace/spectrum) averaged over this aoi fit_backend : str Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used. fit_xvec : ndarray, 1D Values where we have calculated the fit function fit_yvec : ndarray, 1D Values of fit function at solution, for fit_xvec values best_params : ndarray, 1D Best (optimal) fit/model parameters best_sigmas : ndarray, 1D Sigmas (stdev of fit error) at solution best_residual : float Residual vec at solution pguess: tuple of floats Guesses for fit parameters pbounds : 2-tuple, of tuples Bounds for fit, first tuple are the lower bounds, second the upper bounds. fit_options : dict Other options passed to fit(s). Methods    - save_json(filepath) Save into given filepath, ending in '.json'. Arguments     - aoi_num : int Number associated with this AOI sweep_arr : ndarray, 1D Freqs/taus avg_sig_norm : ndarray, 1D sig_norm (trace/spectrum) averaged over this aoi avg_sig : ndarray, 1D sig (trace/spectrum) averaged over this aoi avg_ref : ndarray, 1D ref (trace/spectrum) averaged over this aoi fit_backend : str Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used. fit_xvec : ndarray, 1D Values where we have calculated the fit function fit_yvec : ndarray, 1D Values of fit function at solution, for fit_xvec values best_params : ndarray, 1D Best (optimal) fit/model parameters best_sigmas : ndarray, 1D Sigmas (stdev of fit error) at solution best_residual : float Residual vec at solution pguess: tuple of floats Guesses for fit parameters pbounds : 2-tuple, of tuples Bounds for fit, first tuple are the lower bounds, second the upper bounds. aoi_coords : 4-tuple of ints start_x, start_y, end_x, end_y"
},
{
"ref":"dukit.share.AoiAvgFit.to_dict",
"url":15,
"doc":"Return a dict of params",
"func":1
},
{
"ref":"dukit.share.RoiAvgFit",
"url":15,
"doc":"Holds result from an ROI fit. Only a method to save. Attributes      fit_backend : str Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used. sweep_arr : ndarray, 1D Freqs/taus avg_sig_norm : ndarray, 1D sig_norm (trace/spectrum) averaged over ROI avg_sig : ndarray, 1D sig (trace/spectrum) averaged over ROI avg_ref : ndarray, 1D ref (trace/spectrum) averaged over ROI fit_xvec : ndarray, 1D Values where we have calculated the fit function fit_yvec : ndarray, 1D Values of fit function at solution, for fit_xvec values fit_yvec_guess : ndarray, 1D The fit 'guess' values best_params : ndarray, 1D Best (optimal) fit/model parameters best_sigmas : ndarray, 1D Sigmas (stdev of fit error) at solution best_residual : float Residual vec at solution pguess: tuple of floats Guesses for fit parameters pbounds : 2-tuple, of tuples Bounds for fit, first tuple are the lower bounds, second the upper bounds. Methods    - save_json(filepath) Save into given filepath, ending in '.json'. Arguments     - fit_backend : str Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used. sweep_arr : ndarray, 1D Freqs/taus avg_sig_norm : ndarray, 1D sig_norm (trace/spectrum) averaged over ROI avg_sig : ndarray, 1D sig (trace/spectrum) averaged over ROI avg_ref : ndarray, 1D ref (trace/spectrum) averaged over ROI fit_xvec : ndarray, 1D Values where we have calculated the fit function fit_yvec : ndarray, 1D Values of fit function at solution, for fit_xvec values fit_yvec_guess : ndarray, 1D The fit 'guess' values best_params : ndarray, 1D Best (optimal) fit/model parameters best_sigmas : ndarray, 1D Sigmas (stdev of fit error) at solution best_residual : float Residual vec at solution pguess: tuple of floats Guesses for fit parameters pbounds : 2-tuple, of tuples Bounds for fit, first tuple are the lower bounds, second the upper bounds."
},
{
"ref":"dukit.share.RoiAvgFit.to_dict",
"url":15,
"doc":"Return a dict of params",
"func":1
},
{
"ref":"dukit.warn",
"url":16,
"doc":"Warnings for DUKIT. Always lowest in import heirarchy Functions     - -  duki.warn.warn "
},
{
"ref":"dukit.warn.warn",
"url":16,
"doc":"Throw a custom DUKITWarning with message 'msg'.",
"func":1
},
{
"ref":"dukit.warn.DUKITWarning",
"url":16,
"doc":"allows us to separate dukit warnings from those in other packages."
},
{
"ref":"dukit.itool",
"url":17,
"doc":"This module holds misc image tooling. Functions     - -  dukit.itool.mpl_set_run_config -  dukit.itool.mask_polygons -  dukit.itool.get_im_filtered -  dukit.itool.get_background -  dukit.itool.mu_sigma_inside_polygons -  dukit.itool.plot_image -  dukit.itool.plot_image_on_ax -  dukit.itool.get_colormap_range -  dukit.itool.crop_roi -  dukit.itool.crop_sweep -  dukit.itool.rebin_image_stack -  dukit.itool.smooth_image_stack -  dukit.itool.sum_spatially -  dukit.itool.get_aois -  dukit.itool._iterslice -  dukit.itool._iterframe "
},
{
"ref":"dukit.itool.DEFAULT_RCPARAMS",
"url":17,
"doc":"Set of rcparams we've casually decided are reasonable."
},
{
"ref":"dukit.itool.mpl_set_run_config",
"url":17,
"doc":"Set the matplotlib runtime configuration (rcparams). Parameters      default : bool Use dukit default rcparams? kwargs kwargs to pass to rcparams, see https: matplotlib.org/stable/users/explain/customizing.html",
"func":1
},
{
"ref":"dukit.itool.mask_polygons",
"url":17,
"doc":"Mask image for the given polygon regions. Arguments     - image : 2D array-like Image array to mask. polygons : list, optional List of  qdmpy.shared.polygon.Polygon objects. (the default is None, where image is returned with no mask) invert_mask : bool, optional Invert mask such that background is masked, not polygons (i.e. polygons will be operated on if array is passed to np.mean instead of background). (the default is False) Returns    - masked_im : np.ma.MaskedArray image, now masked",
"func":1
},
{
"ref":"dukit.itool.get_background",
"url":17,
"doc":"Returns a background for given image, via chosen method. Methods available: - \"fix_zero\" - Fix background to be a constant offset (z value) - params required in method_params_dict: \"zero\" an int/float, defining the constant offset of the background - \"three_point\" - Calculate plane background with linear algebra from three [x,y] lateral positions given - params required in method_params_dict: - \"points\" a len-3 iterable containing [x, y] points - \"mean\" - background calculated from mean of image - no params required - \"poly\" - background calculated from polynomial fit to image. - params required in method_params_dict: - \"order\": an int, the 'order' polynomial to fit. (e.g. 1 = plane). - \"gaussian\" - background calculated from _gaussian fit to image (with rotation) - no params required - \"lorentzian\" - as above, but a lorentzian lineshape (with rotation) - \"interpolate\" - Background defined by the dataset smoothed via a sigma-_gaussian filtering, and method-interpolation over masked (polygon) regions. - params required in method_params_dict: - \"interp_method\": nearest, linear, cubic. - \"sigma\": sigma passed to _gaussian filter (see scipy.ndimage._gaussian_filter) which is utilized on the background before interpolating - \"gaussian_filter\" - background calculated from image filtered with a _gaussian filter. - params required in method_params_dict: - \"sigma\": sigma passed to _gaussian filter (see scipy.ndimage._gaussian_filter) - \"gaussian_then_poly\" - runs gaussian then poly subtraction - params required in method_params_dict: - \"order\": an int, the 'order' polynomial to fit. (e.g. 1 = plane). polygon utilization: - if method is not interpolate, the image is masked where the polygons are and the background is calculated without these regions - if the method is interpolate, these regions are interpolated over (and the rest of the image, _gaussian smoothed, is 'background'). Arguments     - image : 2D array-like image to get backgrond of method : str Method to use, available options above  method_params_dict Key-value pairs passed onto each background backend. Required params given above. polygon_nodes : list | None = None Optionally provide polygon nodes. Returns    - im_bground : ndarray 2D numpy array, representing the 'background' of image. mask : ndarray Mask (True pixels were not used to calculate background).",
"func":1
},
{
"ref":"dukit.itool.mu_sigma_inside_polygons",
"url":17,
"doc":"returns (mean, standard_deviation) for image, only _within_ polygon areas.",
"func":1
},
{
"ref":"dukit.itool.get_im_filtered",
"url":17,
"doc":"Wrapped over other filters. Current filters defined: - filter_type = gaussian,  qdmpy.shared.itool._get_im_filtered_gaussian ",
"func":1
},
{
"ref":"dukit.itool.plot_image",
"url":17,
"doc":"Plots an image given by image_data. Only saves figure if path given. Arguments     - image_data : np array, 3D Data that is plot. title : str Title of figure, as well as name for save files c_map : str Colormap object used to map image_data values to a color. c_range : 2-tuple of floats Range of values in image_data to map to colors c_label : str Label for colormap axis opath: str =  , If given saves figure here. show_scalebar: bool = True Show the scalebar(s)? raw_pixel_size: float = float(\"nan\") Pixel size from hardware applied_binning: tuple[int, int] | int = 0, Any additional binning that HAS been applied annotate_polygons: bool = False, Annotate the polygons on image? polygon_nodes: list | None = None, List of polygon nodes, see TODO for format show_tick_marks: bool = False Show tick marks on axes polygon_patch_params: dict | None, default = None Passed to mpl.patches.Polygon. Returns    - fig : matplotlib Figure object ax : matplotlib Axis object",
"func":1
},
{
"ref":"dukit.itool.plot_image_on_ax",
"url":17,
"doc":"Plots an image given by image_data onto given figure and ax. Does not save any data. Arguments     - fig : matplotlib Figure object ax : matplotlib Axis object image_data : np array, 3D Data that is plot. title : str Title of figure, as well as name for save files c_map : str Colormap object used to map image_data values to a color. c_range : 2-tuple of floats Range of values in image_data to map to colors c_label : str Label for colormap axis show_scalebar: bool = True Show the scalebar(s)? raw_pixel_size: float = float(\"nan\") Pixel size from hardware applied_binning: tuple[int, int] | int = 0, Any additional binning that HAS been applied annotate_polygons: bool = False, Annotate the polygons on image? polygon_nodes: list | None = None, List of polygon nodes, see TODO for format show_tick_marks: bool = False Show tick marks on axes polygon_patch_params: dict | None, default = None Passed to mpl.patches.Polygon. Returns    - fig : matplotlib Figure object ax : matplotlib Axis object",
"func":1
},
{
"ref":"dukit.itool.get_colormap_range",
"url":17,
"doc":"Produce a colormap range to plot image from, using the options in c_range_dict. Arguments     - c_range_type : str - \"min_max\" : map between minimum and maximum values in image. - \"deviation_from_mean\" : requires c_range_vals be a 1-tuple float between 0 and 1 'dev'. Maps between (1 - dev)  mean and (1 + dev)  mean. - \"min_max_symmetric_about_mean\" : map symmetrically about mean, capturing all values in image (default). - \"min_max_symmetric_about_zero\" : map symmetrically about zero, capturing all values in image. - \"percentile\" : requires c_range_vals be a 2-tuple w vals between 0 and 100. Maps the range between those percentiles of the data. - \"percentile_symmetric_about_zero\" : requires c_range_vals be a 2-tuple w vals between 0 and 100. Maps symmetrically about zero, capturing all values between those percentiles in the data (plus perhaps a bit more to ensure symmety) - \"strict_range\" : requiresc_range_vals 2-tuple. Maps colors directly between the values given. - \"mean_plus_minus\" : mean plus or minus this value. c_range_vals must be a 1-tuple float c_range_vals : tuple See above for allowed options auto_sym_zero : bool, default=True Try and make symmetric around zero, if logical? image : array-like 2D array (image) that fn returns colormap range for. Returns    - c_range : tuple length 2 i.e. [min value to map to a color, max value to map to a color]",
"func":1
},
{
"ref":"dukit.itool.crop_roi",
"url":17,
"doc":"Parameters      seq : array-like Image or image-stack you want to crop roi_coords: 4-tuple start_x: int, start_y: int, end_x: int, end_y: int If any are -1, then it sets to the edge of image. Returns    - seq_cropped : np.ndarray Image or image-stack cropped.",
"func":1
},
{
"ref":"dukit.itool.crop_sweep",
"url":17,
"doc":"Crop spectral dimension. Usually used to remove first one/few points of e.g. ODMR. Parameters      sweep_arr, sig, ref, sig_norm : npdarray's ndarrays to be cropped in spectral dimension. (sweep_arr only has spec dim). rem_start, rem_end : int How many pts to remove from start & end of spectral dimension. Returns    - sweep_arr, sig, ref, sig_norm : npdarray's All as input, put with spectral dimension cropped to arr[rem_start:-rem_end]",
"func":1
},
{
"ref":"dukit.itool.smooth_image_stack",
"url":17,
"doc":"Smooth image stack in spatial dimensions with gaussian. Parameters      stack : ndarray sigma : 2-tuple of floats or float If float, then symmetric smoothing in each dim, otherwise tuple(x, y) truncate : float, default=4.0 Truncate the filter at this many standard deviations. See scipy.ndimage.gaussian_filter. Returns    - smooth_stack : ndarray Image stack smoothed in spatial dimensions.",
"func":1
},
{
"ref":"dukit.itool.rebin_image_stack",
"url":17,
"doc":"Rebin image stack in spatial dimensions. Parameters      stack : ndarray additional_bins : 2-tuple of ints, or int Binning in-plane. Make it a power of 2. Binning in x then y if 2-tuple, else symmetric. Returns    - smooth_stack : ndarray Image stack smoothed in spatial dimensions.",
"func":1
},
{
"ref":"dukit.itool.sum_spatially",
"url":17,
"doc":"Sum over 0th (spectral) dim of seq if 3D, else return as is.",
"func":1
},
{
"ref":"dukit.itool.get_aois",
"url":17,
"doc":"",
"func":1
},
{
"ref":"dukit.itool._iterslice",
"url":17,
"doc":"Iterate through array x in slices along axis 'axis' (defaults to 0). E.g. iterslice(shape(y,x,freqs), axis=-1) will give iter. of 1d freq slices.",
"func":1
},
{
"ref":"dukit.itool._iterframe",
"url":17,
"doc":"iterframe(shape(y,x,freqs will give iter. of 2d y,x frames.",
"func":1
},
{
"ref":"dukit.polygon",
"url":18,
"doc":"This module holds the Polygon class. A class to compute if a point lies inside/outside/on-side of a polygon. Also defined is a function (polygon_selector) that can be called to select a polygon region on an image. For use check examples. Apologies for lack of typing, I don't understand sufficiently and it currently  just works . Polygon    - This is a Python 3 implementation of the Sloan's improved version of the Nordbeck and Rystedt algorithm, published in the paper: SLOAN, S.W. (1985): A point-in-polygon program. Adv. Eng. Software, Vol 7, No. 1, pp 45-47. This class has 1 method (is_inside) that returns the minimum distance to the nearest point of the polygon: If is_inside  0 then point is inside the polygon. Sam Scholten copied from: http: code.activestate.com/recipes/578381-a-point-in-polygon-program-sw-sloan-algorithm/ -> swapped x & y args order (etc.) for image use. Classes    - -  dukit.polygon.Polygon Functions     - -  dukit.polygon.load_polygon_nodes -  dukit.polygon.polygon_selector -  dukit.polygon.PolygonSelectionWidget "
},
{
"ref":"dukit.polygon.Polygon",
"url":18,
"doc":"Polygon object. Arguments     - y : array-like A sequence of nodal y-coords (all unique). x : array-like A sequence of nodal x-coords (all unique)."
},
{
"ref":"dukit.polygon.Polygon.get_nodes",
"url":18,
"doc":"",
"func":1
},
{
"ref":"dukit.polygon.Polygon.get_yx",
"url":18,
"doc":"",
"func":1
},
{
"ref":"dukit.polygon.Polygon.is_inside",
"url":18,
"doc":"",
"func":1
},
{
"ref":"dukit.polygon.polygon_selector",
"url":18,
"doc":"Generates mpl (qt) gui for selecting a polygon. NOTE: you probably just want to use PolygonSelectionWidget. Arguments     - array : path OR arraylike Path to (numpy) .txt file to load as image. OR can be an arraylike directly json_output_path : str or path-like, default=\"~/poly.json\" Path to put output json, defaults to home/poly.json. json_input_path : str or path-like, default=None Loads previous polygons at this path for editing. mean_plus_minus : float, default=None Plot image with color scaled to mean +- this number. strict_range: length 2 list, default=None Plot image with color scaled between these values. Precedence over mean_plus_minus. print_help : bool, default=False View this message. pad : bool If > 0, pads with zeros by 'pad' fraction times the image size in both dimensions The 'padder' (see  qdmpy.sharead.fourier.unpad_image ) is placed in the output dict/json.  kwargs : dict Other keyword arguments to pass to plotters. Currently implemented: cmap : string Passed to imshow. lineprops : dict Passed to PolygonSelectionWidget. markerprops : dict Passed to PolygonSelectionWidget. GUI help     In the mpl gui, select points to draw polygons. Press 'enter' to continue in the program. Press the 'esc' key to reset the current polygon Hold 'shift' to move all of the vertices (from all polygons) Hold 'r' and scroll to resize all of the polygons. 'ctrl' to move a single vertex in the current polygon 'alt' to start a new polygon (and finalise the current one) 'del' to clear all lines from the graphic (thus deleting all polygons). 'right click' on a vertex (of a finished polygon) to remove it.",
"func":1
},
{
"ref":"dukit.polygon.PolygonSelectionWidget",
"url":18,
"doc":"How to Use      selector = PolygonSelectionWidget(ax,  .) plt.show() selector.disconnect() polygon_lst = selector.get_polygon_lst()   GUI help     In the mpl gui, select points to draw polygons. Press 'enter' to continue in the program. Press the 'esc' key to reset the current polygon Hold 'shift' to move all of the vertices (from all polygons) Hold 'r' and scroll to resize all of the polygons. 'ctrl' to move a single vertex in the current polygon 'alt' to start a new polygon (and finalise the current one) 'del' to clear all lines from the graphic (thus deleting all polygons). 'right click' on a vertex (of a finished polygon) to remove it.  "
},
{
"ref":"dukit.polygon.PolygonSelectionWidget.onselect",
"url":18,
"doc":"",
"func":1
},
{
"ref":"dukit.polygon.PolygonSelectionWidget.disconnect",
"url":18,
"doc":"",
"func":1
},
{
"ref":"dukit.polygon.PolygonSelectionWidget.get_polygons_lst",
"url":18,
"doc":"",
"func":1
},
{
"ref":"dukit.polygon.PolygonSelectionWidget.load_nodes",
"url":18,
"doc":"",
"func":1
},
{
"ref":"dukit.polygon.load_polygon_nodes",
"url":18,
"doc":"Loads polygon nodes from json file. Arguments     - poly_path_or_dict : str | dict Path to json or pickle/dill file containing polygon nodes, or directly as a dict Returns    - list[npt.NDArray] List of polygons, each polygon is an array of nodes.",
"func":1
},
{
"ref":"dukit.systems",
"url":19,
"doc":"This sub-package holds classes and functions to define (microscope) systems. Classes    - -  dukit.systems.System -  dukit.systems.MelbSystem -  dukit.systems.LVControl -  dukit.systems.PyControl -  dukit.systems.Zyla -  dukit.systems.CryoWidefield -  dukit.systems.LegacyCryoWidefield -  dukit.systems.Argus -  dukit.systems.LegacyArgus -  dukit.systems.PyCryoWidefield "
},
{
"ref":"dukit.systems.System",
"url":19,
"doc":"Abstract class defining what is expected for a system. Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.System.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.System.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.System.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.System.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.System.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.System.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.System.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.systems.MelbSystem",
"url":19,
"doc":"Some shared methods for melbourne systems. Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.MelbSystem.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.MelbSystem.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.MelbSystem.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.MelbSystem.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.MelbSystem.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.MelbSystem.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.MelbSystem.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.systems.LVControl",
"url":19,
"doc":"Older Labview-based control software, save formats etc. Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.LVControl.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.LVControl.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.LVControl.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.LVControl.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.LVControl.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.LVControl.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.LVControl.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.systems.PyControl",
"url":19,
"doc":"Newer systems, using python control software -> new API Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.PyControl.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.PyControl.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.PyControl.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.PyControl.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.PyControl.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.PyControl.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.PyControl.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.systems.Zyla",
"url":19,
"doc":"Specific system details for the Zyla QDM. Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.Zyla.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.Zyla.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.Zyla.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.Zyla.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.Zyla.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.Zyla.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.Zyla.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.systems.CryoWidefield",
"url":19,
"doc":"Specific system details for Cryogenic (Attocube) widefield QDM. Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.CryoWidefield.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.CryoWidefield.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.CryoWidefield.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.CryoWidefield.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.CryoWidefield.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.CryoWidefield.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.CryoWidefield.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.systems.LegacyCryoWidefield",
"url":19,
"doc":"Specific system details for cryogenic (Attocube) widefield QDM. - Legacy binning version Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.LegacyCryoWidefield.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.LegacyCryoWidefield.determine_binning",
"url":19,
"doc":"silly old binning convention -> change when labview updated to new binning",
"func":1
},
{
"ref":"dukit.systems.LegacyCryoWidefield.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.LegacyCryoWidefield.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.LegacyCryoWidefield.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.LegacyCryoWidefield.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.LegacyCryoWidefield.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.LegacyCryoWidefield.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.systems.Argus",
"url":19,
"doc":"Specific system details for Argus room-temperature widefield QDM. Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.Argus.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.Argus.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.Argus.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.Argus.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.Argus.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.Argus.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.Argus.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.systems.LegacyArgus",
"url":19,
"doc":"System for Argus with old binning convention Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.LegacyArgus.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.LegacyArgus.determine_binning",
"url":19,
"doc":"Silly old binning convention -> change when labview updated to new binning",
"func":1
},
{
"ref":"dukit.systems.LegacyArgus.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.LegacyArgus.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.LegacyArgus.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.LegacyArgus.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.LegacyArgus.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.LegacyArgus.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.systems.PyCryoWidefield",
"url":19,
"doc":"Specific system details for Cryogenic (Attocube) widefield QDM. Initialize the  System , optionally supply known bias field and override the  System -defined microscope distances/etc. Parameters      pixel_size: float | None, default=None Define pixel size manually. If set, below 4 params are disregarded. Either pixel_size or the below 4 must be defined either in init or hard-coded into System (sub-)class. Given in metres. sensor_pixel_pitch: float | None, default=None Effective pixel 'size' at camera sensor. Given in metres. obj_mag: float | None, default=None Magnification of objective used. Usually you will supply an int. obj_ref_focal_length: float | None, default=None Reference focal length from objective manufacturer (in metres). 200e-3 for Nikon or Leica. For Olympus use 180e-3, for Zeiss 165e-3. camera_tube_lens: float | None, default=None Tube lens length (m) used to focus light onto camera. bias_mag: float | None, default=None Magnitude of bias field (if known, else None) in Teslas. bias_theta: float | None, default=None Polar angle (deg) of bias field. bias_phi: float | None, default=None Azimuthal angle (deg) of bias field."
},
{
"ref":"dukit.systems.PyCryoWidefield.name",
"url":19,
"doc":"Name of the system."
},
{
"ref":"dukit.systems.PyCryoWidefield.read_image",
"url":19,
"doc":"Method that must be defined to read raw data in from filepath. Parameters      filepath : str or Pathlib etc. object Path to measurement file ignore_ref : bool Ignore any reference measurements. (i.e. no-RF lock-in) norm : str Normalisation method. \"div\", \"sub\" or \"true_sub\". (latter for T1 datasets) Returns    - sig : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. ref : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. sig_norm : np array, 3D Format: [y, x, sweep_vals]. Not cropped etc. Notes   - if norm  \"sub\": sig_norm = 1 + (sig - ref) / (sig + ref) elif norm  \"div\": sig_norm = sig / ref elif norm  \"true_sub\": sig_norm = (sig - ref) / np.nanmax(sig - ref).reshape(sig.shape[:-1]+(1, ",
"func":1
},
{
"ref":"dukit.systems.PyCryoWidefield.read_sweep_arr",
"url":19,
"doc":"Method that must be defined to read sweep_arr in from filepath. Arguments     - filepath : str or Pathlib etc. object Path to measurement file Returns    - sweep_arr : np array, 1D List of sweep value, either freq (MHz) or taus (s).",
"func":1
},
{
"ref":"dukit.systems.PyCryoWidefield.get_hardware_binning",
"url":19,
"doc":"Method that must be defined to define the camera binning from metadata Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.PyCryoWidefield.get_bias_field",
"url":19,
"doc":"Method to get magnet bias field from experiment metadata, i.e. if set with programmable electromagnet. Default: False, (None, None, None). Arguments     - filepath : str or Pathlib etc. object Path to measurement file auto_read : bool, default=False Read from metadata? Returns    - bias_on : bool Was programmable bias field used? bias_field : tuple Tuple representing vector bias field (B_mag (Tesla), B_theta (rad), B_phi (rad ",
"func":1
},
{
"ref":"dukit.systems.PyCryoWidefield.get_raw_pixel_size",
"url":19,
"doc":"Get raw (from camera, without additional binning) pixel size. Arguments     - filepath : str or Pathlib etc. object Path to measurement file",
"func":1
},
{
"ref":"dukit.systems.PyCryoWidefield.norm",
"url":19,
"doc":"Parameters      sig : npt.NDArray signal ref : npt.NDArray reference norm : str = \"div\" normalisation method in [\"div\", \"sub\", \"true_sub\"] Returns    - sig_norm : npt.NDArray normalised signal",
"func":1
},
{
"ref":"dukit.widget",
"url":20,
"doc":"Various GUI widgets for use in the DUKIT. Not very well documented. But I don't really understand it so that's fine. How to use? See polygon.py and linecut.py Don't import any other dukit modules, ensure this is a leaf in the dep. tree. Apologies for lack of typing, I don't understand this stuff sufficiently. But it does currently  just work . Classes    - -  dukit.widget.PolygonSelector -  dukit.widget.LineSelector -  dukit.widget.LinecutSelectionWidget -  dukit.widget.BulkLinecutWidget "
},
{
"ref":"dukit.widget.Widget",
"url":20,
"doc":"Abstract base class for GUI neutral widgets"
},
{
"ref":"dukit.widget.Widget.drawon",
"url":20,
"doc":""
},
{
"ref":"dukit.widget.Widget.eventson",
"url":20,
"doc":""
},
{
"ref":"dukit.widget.Widget.set_active",
"url":20,
"doc":"Set whether the widget is active.",
"func":1
},
{
"ref":"dukit.widget.Widget.get_active",
"url":20,
"doc":"Get whether the widget is active.",
"func":1
},
{
"ref":"dukit.widget.Widget.active",
"url":20,
"doc":"Is the widget active?"
},
{
"ref":"dukit.widget.Widget.ignore",
"url":20,
"doc":"Return True if event should be ignored. This method (or a version of it) should be called at the beginning of any event callback.",
"func":1
},
{
"ref":"dukit.widget.AxesWidget",
"url":20,
"doc":"Widget that is connected to a single :class: ~matplotlib.axes.Axes . To guarantee that the widget remains responsive and not garbage-collected, a reference to the object should be maintained by the user. This is necessary because the callback registry maintains only weak-refs to the functions, which are member functions of the widget. If there are no references to the widget object it may be garbage collected which will disconnect the callbacks. Attributes:  ax : :class: ~matplotlib.axes.Axes The parent axes for the widget  canvas : :class: ~matplotlib.backend_bases.FigureCanvasBase subclass The parent figure canvas for the widget.  active : bool If False, the widget does not respond to events."
},
{
"ref":"dukit.widget.AxesWidget.connect_event",
"url":20,
"doc":"Connect callback with an event. This should be used in lieu of  figure.canvas.mpl_connect since this function stores callback ids for later clean up.",
"func":1
},
{
"ref":"dukit.widget.AxesWidget.disconnect_events",
"url":20,
"doc":"Disconnect all events created by this widget.",
"func":1
},
{
"ref":"dukit.widget.AxesWidget.set_active",
"url":20,
"doc":"Set whether the widget is active.",
"func":1
},
{
"ref":"dukit.widget.AxesWidget.get_active",
"url":20,
"doc":"Get whether the widget is active.",
"func":1
},
{
"ref":"dukit.widget.AxesWidget.active",
"url":20,
"doc":"Is the widget active?"
},
{
"ref":"dukit.widget.AxesWidget.ignore",
"url":20,
"doc":"Return True if event should be ignored. This method (or a version of it) should be called at the beginning of any event callback.",
"func":1
},
{
"ref":"dukit.widget.ToolHandles",
"url":20,
"doc":"Control handles for canvas tools. Arguments     - ax : :class: matplotlib.axes.Axes Matplotlib axes where tool handles are displayed. x, y : 1D arrays Coordinates of control handles. marker : str Shape of marker used to display handle. See  matplotlib.pyplot.plot . marker_props : dict Additional marker properties. See :class: matplotlib.lines.Line2D ."
},
{
"ref":"dukit.widget.ToolHandles.x",
"url":20,
"doc":""
},
{
"ref":"dukit.widget.ToolHandles.y",
"url":20,
"doc":""
},
{
"ref":"dukit.widget.ToolHandles.set_data",
"url":20,
"doc":"Set x and y positions of handles",
"func":1
},
{
"ref":"dukit.widget.ToolHandles.set_visible",
"url":20,
"doc":"",
"func":1
},
{
"ref":"dukit.widget.ToolHandles.set_animated",
"url":20,
"doc":"",
"func":1
},
{
"ref":"dukit.widget.ToolHandles.closest",
"url":20,
"doc":"Return index and pixel distance to closest index.",
"func":1
},
{
"ref":"dukit.widget.PolygonSelector",
"url":20,
"doc":"OLD DOCSTRING Select a polygon region of an axes. Place vertices with each mouse click, and make the selection by completing the polygon (clicking on the first vertex). Hold the  ctrl key and click and drag a vertex to reposition it (the  ctrl key is not necessary if the polygon has already been completed). Hold the  shift key and click and drag anywhere in the axes to move all vertices. Press the  esc key to start a new polygon. For the selector to remain responsive you must keep a reference to it. Arguments     - ax : :class: ~matplotlib.axes.Axes The parent axes for the widget. onselect : function When a polygon is completed or modified after completion, the  onselect function is called and passed a list of the vertices as  (xdata, ydata) tuples. useblit : bool, optional lineprops : dict, optional The line for the sides of the polygon is drawn with the properties given by  lineprops . The default is  dict(color='k', linestyle='-', linewidth=2, alpha=0.5) . markerprops : dict, optional The markers for the vertices of the polygon are drawn with the properties given by  markerprops . The default is  dict(marker='o', markersize=7, mec='k', mfc='k', alpha=0.5) . vertex_select_radius : float, optional A vertex is selected (to complete the polygon or to move a vertex) if the mouse click is within  vertex_select_radius pixels of the vertex. The default radius is 15 pixels. Examples     :doc: /gallery/widgets/polygon_selector_demo "
},
{
"ref":"dukit.widget.PolygonSelector.onmove",
"url":20,
"doc":"Cursor move event handler and validator",
"func":1
},
{
"ref":"dukit.widget.PolygonSelector.draw_polygon",
"url":20,
"doc":"Redraw the polygon(s) based on the new vertex positions.",
"func":1
},
{
"ref":"dukit.widget.PolygonSelector.verts",
"url":20,
"doc":"Get the polygon vertices. Returns    - list A list of the vertices of the polygon as  (xdata, ydata) tuples. for each polygon (A, B,  .) selected  [ [(Ax1, Ay1), (Ax2, Ay2)], [(Bx1, By1), (Bx2, By2)] ] "
},
{
"ref":"dukit.widget.PolygonSelector.xy_verts",
"url":20,
"doc":"Return list of the vertices for each polygon in the format: [ ( [Ax1, Ax2,  .], [Ay1, Ay2,  .] ), ( [Bx1, Bx2,  .], [By1, By2,  .] ) ]"
},
{
"ref":"dukit.widget.PolygonSelector.connect_event",
"url":20,
"doc":"Connect callback with an event. This should be used in lieu of  figure.canvas.mpl_connect since this function stores callback ids for later clean up.",
"func":1
},
{
"ref":"dukit.widget.PolygonSelector.disconnect_events",
"url":20,
"doc":"Disconnect all events created by this widget.",
"func":1
},
{
"ref":"dukit.widget.PolygonSelector.set_active",
"url":20,
"doc":"Set whether the widget is active.",
"func":1
},
{
"ref":"dukit.widget.PolygonSelector.get_active",
"url":20,
"doc":"Get whether the widget is active.",
"func":1
},
{
"ref":"dukit.widget.PolygonSelector.active",
"url":20,
"doc":"Is the widget active?"
},
{
"ref":"dukit.widget.PolygonSelector.ignore",
"url":20,
"doc":"Return True if event should be ignored. This method (or a version of it) should be called at the beginning of any event callback.",
"func":1
},
{
"ref":"dukit.widget.LineSelector",
"url":20,
"doc":"similar to PolygonSelector but an open line."
},
{
"ref":"dukit.widget.LineSelector.onmove",
"url":20,
"doc":"Cursor move event handler and validator",
"func":1
},
{
"ref":"dukit.widget.LineSelector.draw_line",
"url":20,
"doc":"Redraw the line based on the new vertex positions.",
"func":1
},
{
"ref":"dukit.widget.LineSelector.verts",
"url":20,
"doc":"Get the line vertices. Returns    - list A list of the vertices of the line as  (xdata, ydata) tuples.  [(Ax1, Ay1), (Ax2, Ay2)] "
},
{
"ref":"dukit.widget.LineSelector.current_verts",
"url":20,
"doc":""
},
{
"ref":"dukit.widget.LineSelector.xy_verts",
"url":20,
"doc":"Return list of the vertices for the line in this format: ( [Ax1, Ax2,  .], [Ay1, Ay2,  .] )"
},
{
"ref":"dukit.widget.LineSelector.connect_event",
"url":20,
"doc":"Connect callback with an event. This should be used in lieu of  figure.canvas.mpl_connect since this function stores callback ids for later clean up.",
"func":1
},
{
"ref":"dukit.widget.LineSelector.disconnect_events",
"url":20,
"doc":"Disconnect all events created by this widget.",
"func":1
},
{
"ref":"dukit.widget.LineSelector.set_active",
"url":20,
"doc":"Set whether the widget is active.",
"func":1
},
{
"ref":"dukit.widget.LineSelector.get_active",
"url":20,
"doc":"Get whether the widget is active.",
"func":1
},
{
"ref":"dukit.widget.LineSelector.active",
"url":20,
"doc":"Is the widget active?"
},
{
"ref":"dukit.widget.LineSelector.ignore",
"url":20,
"doc":"Return True if event should be ignored. This method (or a version of it) should be called at the beginning of any event callback.",
"func":1
},
{
"ref":"dukit.widget.BulkLinecutWidget",
"url":20,
"doc":"How to use      import matplotlib.pyplot as plt import numpy as np from qdmpy.shared.linecut import BulkLinecutWidget  TODO update path = \" \" times = [0.325, 1, 5, 10, 20, 21, 22, 25, 30, 40] paths = [f\"{path}/{t}.txt\" for t in times] images = [np.loadtxt(p) for p in paths] selector_image = images[4] fig, axs = plt.subplots(ncols=3, figsize=(12, 6 axs[0].imshow(selector_image)  (data can be nans if you want an empty selector) selector = BulkLinecutWidget( axs, images, times) plt.show() selector.disconnect(path=\"/home/samsc/share/result.json\")"
},
{
"ref":"dukit.widget.BulkLinecutWidget.ondraw",
"url":20,
"doc":"",
"func":1
},
{
"ref":"dukit.widget.BulkLinecutWidget.onselect",
"url":20,
"doc":"",
"func":1
},
{
"ref":"dukit.widget.BulkLinecutWidget.disconnect",
"url":20,
"doc":"",
"func":1
},
{
"ref":"dukit.widget.LinecutSelectionWidget",
"url":20,
"doc":"How to Use      fig, axs = plt.subplots(ncols=2) axs[0].imshow(data)  (data may be nans if you want empty selector) selector = LinecutSelectionWidget(axs[0], axs[1],  .) plt.show() selector.disconnect()"
},
{
"ref":"dukit.widget.LinecutSelectionWidget.ondraw",
"url":20,
"doc":"",
"func":1
},
{
"ref":"dukit.widget.LinecutSelectionWidget.onselect",
"url":20,
"doc":"",
"func":1
},
{
"ref":"dukit.widget.LinecutSelectionWidget.disconnect",
"url":20,
"doc":"",
"func":1
},
{
"ref":"dukit.plot",
"url":21,
"doc":"This module holds functions for plotting. Functions     - -  dukit.plot.roi_pl_image -  dukit.plot.aoi_pl_image -  dukit.plot.roi_avg_fits -  dukit.plot.aoi_spectra -  dukit.plot.aoi_spectra_fit -  dukit.plot.pl_param_image -  dukit.plot.pl_param_images -  dukit.plot._add_patch_rect "
},
{
"ref":"dukit.plot.roi_pl_image",
"url":21,
"doc":"Plots full pl image with ROI region annotated. Arguments     - pl_image : np array, 2D UNCCROPPED pl image, but binned roi_coords : tuple[int, int, int, int] ROI coordinates c_range_type : str =  Type of colormap range to use. See  dukit.itool.get_colormap_range c_range_values : tuple[float, float] = () Values to use for colormap range. See  dukit.itool.get_colormap_range opath : str =  If supplied, saves figure here.  kwargs Plotting options passed to  dukit.itool.plot_image_on_ax Returns    - fig : plt.Figure ax : plt.Axes",
"func":1
},
{
"ref":"dukit.plot.aoi_pl_image",
"url":21,
"doc":"Plots pl image cut down to ROI, with annotated AOI regions. Arguments     - pl_image : np array, 2D pl image AFTER cropping.  aoi_coords : tuple[int, int, int, int] AOI coordinates c_range_type : str =  Type of colormap range to use. See  dukit.itool.get_colormap_range c_range_values : tuple[float, float] = () Values to use for colormap range. See  dukit.itool.get_colormap_range opath : str If supplied, saves figure here.  kwargs Plotting options passed to  dukit.itool.plot_image_on_ax Returns    - fig : plt.Figure ax : plt.Axes",
"func":1
},
{
"ref":"dukit.plot.roi_avg_fits",
"url":21,
"doc":"Plots fit of spectrum averaged across ROI, as well as corresponding residual values. Arguments     - roi_results : dict[str, dukit.share.RoiAvgFit dict[\"fit_backend\"] => RoiAvgFit opath : str If given, save figure here. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"dukit.plot.aoi_spectra",
"url":21,
"doc":"Plots spectra from each AOI, as well as subtraction and division norms. Arguments     - sig : np array, 3D Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] ref : np array, 3D Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed. Cut down to ROI. Format: [sweep_vals, y, x] sweep_arr : ndarray List of sweep parameter values (with removed unwanted sweeps at start/end) specpath : str Path (preferably to json) to save spectra in. Note you  probably want to use TODO(fit_aois) to output instead. opath : str Path to save figure in. Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"dukit.plot.aoi_spectra_fit",
"url":21,
"doc":"Plots sig and ref spectra, sub and div normalisation and fit for the ROI average, a single pixel, and each of the AOIs. Stacked on top of each other for comparison. The ROI average fit is plot against the fit of all of the others for comparison. Note here and elsewhere the single pixel check is the first element of the AOI array. Arguments     - aoi_fit_results : dict dict[\"AOI_num\"] => dict[\"fit_backend\"] => dukit.share.AoiFit roi_fit_results : dict dict[\"fit_backend\"] => dukit.share.RoiAvgFit aoi_coords : tuple[int, int, int, int] AOI coordinates img_shape : tuple[int, int] Shape of the image, used to get the single pixel AOI. opath : str If given, save figure here. Returns    - fig : plt.Figure ax : plt.Axes",
"func":1
},
{
"ref":"dukit.plot.pl_param_image",
"url":21,
"doc":"Plots an image corresponding to a single parameter in pixel_fit_params. Arguments     - options : dict Generic options dict holding all the user options. fit_model : dukit.pl.model.FitModel Model we're fitting to. pixel_fit_params : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. param_name : str Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'. Optional arguments          c_range_type : str =  Type of colormap range to use. See  dukit.itool.get_colormap_range c_range_values : tuple[float, float] = () Values to use for colormap range. See  dukit.itool.get_colormap_range param_number : int, default-0 Which version of the parameter you want. I.e. there might be 8 independent parameters in the fit model called 'pos', each labeled 'pos_0', 'pos_1' etc. errorplot : bool Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so ylabel & save names are changed accordingly. Can't be True if param_name='residual'. opath : str If given, tries to save figure here.  kwargs Plotting options passed to  dukit.itool.plot_image_on_ax Returns    - fig : matplotlib Figure object",
"func":1
},
{
"ref":"dukit.plot.pl_param_images",
"url":21,
"doc":"Plots images for all independent versions of a single parameter type in pixel_fit_params. Arguments     - fit_model :  qdmpy.pl.model.FitModel Model we're fitting to. pixel_fit_params : dict Dictionary, key: param_keys, val: image (2D) of param values across FOV. param_name : str Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'. c_range_type : str =  Type of colormap range to use. See  dukit.itool.get_colormap_range c_range_values : tuple[float, float] = () Values to use for colormap range. See  dukit.itool.get_colormap_range errorplot : bool Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so ylabel & save names are changed accordingly. Can't be True if param_name='residual'. opath : str If given, tries to save figure here.  kwargs Plotting options passed to  dukit.itool.plot_image_on_ax Returns    - fig : plt.Figure ax : plt.Axes",
"func":1
},
{
"ref":"dukit.plot._add_patch_rect",
"url":21,
"doc":"Adds a rectangular annotation onto ax. Arguments     - ax : matplotlib Axis object aoi_coord: tuple[int, int, int, int] start_x, start_y, end_x, end_y for aoi/roi Optional arguments          label : str Text to label annotated square with. Color is defined by edgecolor. Default: None. edgecolor : str Color of label and edge of annotation. Default: \"b\".",
"func":1
},
{
"ref":"dukit.plot.b_defects",
"url":21,
"doc":"Plots the b_defects. Parameters      b_defects : tuple[npt.ArrayLike] b_defect images or values. name : str =  title etc. c_range_type : str =  Type of colormap range to use. See  dukit.itool.get_colormap_range c_range_values : tuple[float, float] = () Values to use for colormap range. See  dukit.itool.get_colormap_range opath : str =  If given, save figure here.  kwargs Plotting options passed to  dukit.itool.plot_image_on_ax Returns    - fig : plt.Figure ax : plt.Axes",
"func":1
},
{
"ref":"dukit.plot.dshifts",
"url":21,
"doc":"Plots the b_defects. Parameters      dshifts : tuple[npt.ArrayLike] b_defect images or values. name : str =  title etc. c_range_type : str =  Type of colormap range to use. See  dukit.itool.get_colormap_range c_range_values : tuple[float, float] = () Values to use for colormap range. See  dukit.itool.get_colormap_range opath : str =  If given, save figure here.  kwargs Plotting options passed to  dukit.itool.plot_image_on_ax Returns    - fig : plt.Figure ax : plt.Axes",
"func":1
}
]