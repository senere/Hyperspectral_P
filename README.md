# Hyperspectral Data Preprocessing Pipeline

A comprehensive Python implementation for processing hyperspectral remote sensing data, from raw acquisition to advanced spectral analysis.

## Project Overview

This project provides an end-to-end hyperspectral data preprocessing workflow implemented in a Jupyter Notebook. It includes modules for data acquisition, radiometric and atmospheric corrections, geometric transformations, dimensionality reduction, spectral unmixing, and anomaly detection.

The pipeline is designed to handle hyperspectral imagery from sources such as:
- AVIRIS (Airborne Visible/Infrared Imaging Spectrometer)
- Custom hyperspectral sensors

## Key Features

- **Data Acquisition Module**: Download and validate hyperspectral data from USGS EarthExplorer
- **Radiometric Correction**: Digital Number (DN) to Radiance conversion with sensor-specific calibration
- **Atmospheric Correction**: QUAC and FLAASH algorithms for surface reflectance retrieval
- **Geometric Correction**: Orthorectification and georeferencing
- **Dimensionality Reduction**: Minimum Noise Fraction (MNF) and Principal Component Analysis (PCA)
- **Band Selection**: Entropy, variance, and correlation-based methods
- **Spectral Unmixing**: Linear spectral unmixing with Pixel Purity Index (PPI) endmember extraction
- **Spectral Indices**: NDVI, NDBI, NDMI, and custom index calculation
- **Anomaly Detection**: RX detector and Spectral Angle Mapper (SAM) implementations
- **Visualization**: Multi-spectral RGB composition, band statistics, and spectral profiles

## Project Structure

```
Hyperspectral_P/
├── data_preprocessing.ipynb          # Main Jupyter notebook
├── README.md                         # This file
├── data/                             # Input hyperspectral data directory
│   ├── raw/                          # Raw satellite imagery
│   ├── processed/                    # Processed data outputs
│   ├── radiometric_correction/       # Radiometrically corrected images
│   ├── atmospheric_correction/       # Atmospherically corrected images
│   ├── geometric_correction/         # Geometrically corrected images
│   └── dimensionality_reduction/     # MNF/PCA transformed data
├── outputs/                          # Model outputs and exports
│   ├── unmixing_results/             # Spectral unmixing abundances
│   ├── anomaly_maps/                 # Anomaly detection maps
│   └── statistics/                   # Statistical analysis results
└── figures/                          # Generated visualizations and plots
    ├── rgb_compositions/             # Multi-spectral RGB images
    ├── band_statistics/              # Band-wise statistics plots
    └── spectral_analysis/            # Spectral profile plots
```

## Modules and Classes

### 1. DataAcquisition
Handles hyperspectral data download and organization.

**Key Methods:**
- `download_usgs_data()`: Download data from USGS EarthExplorer
- `validate_metadata()`: Extract and validate image metadata
- `organize_data()`: Create standardized directory structure

**Example:**
```python
acq = DataAcquisition()
metadata = acq.validate_metadata(DATA_DIR / 'sample.tif')
acq.organize_data(raw_dir, organized_dir)
```

### 2. RadiometricCorrection
Converts Digital Numbers (DN) to spectral radiance using sensor calibration parameters.

**Key Methods:**
- `dn_to_radiance()`: DN to radiance conversion using gain and offset
- `thermal_to_brightness_temp()`: Convert thermal bands to brightness temperature
- `remove_striping()`: Remove sensor striping artifacts

**Example:**
```python
rad_corrector = RadiometricCorrection(sensor_gain, sensor_offset)
radiance = rad_corrector.dn_to_radiance(dn_image)
```

### 3. AtmosphericCorrection
Removes atmospheric effects to retrieve surface reflectance.

**Key Methods:**
- `quac_simple()`: Quick Atmospheric Correction (dark object subtraction)
- `flaash_simple()`: Fast Line-of-sight Atmospheric Analysis
- `spectral_angle_mapper_correction()`: Spectral similarity using SAM
- `apply_quality_mask()`: Remove cloudy and water pixels

**Example:**
```python
atm_corrector = AtmosphericCorrection()
surface_ref = atm_corrector.quac_simple(toa_reflectance)
```

### 4. DimensionalityReduction
Reduces the dimensionality of hyperspectral data while preserving information.

**Key Methods:**
- `compute_mnf()`: Minimum Noise Fraction transformation
- `compute_pca()`: Principal Component Analysis
- `select_bands()`: Intelligent band selection

**Methods:**
- `entropy`: Information entropy-based selection
- `variance`: Variance-based selection
- `correlation`: Decorrelation-based selection

**Example:**
```python
dimred = DimensionalityReduction()
mnf_result, eigenvalues = dimred.compute_mnf(image, n_components=10)
selected_bands, indices = dimred.select_bands(image, method='entropy', n_bands=15)
```

### 5. SpectralUnmixing
Decomposes mixed pixels into pure endmember signatures and abundance maps.

**Key Methods:**
- `linear_spectral_unmixing()`: Least squares unmixing
- `extract_endmembers_ppi()`: Pixel Purity Index for endmember extraction
- `spectral_indices()`: Calculate NDVI, NDBI, NDMI, etc.

**Example:**
```python
unmixing = SpectralUnmixing()
endmembers = unmixing.extract_endmembers_ppi(image, n_endmembers=5)
abundances = unmixing.linear_spectral_unmixing(image, endmembers)
```

### 6. AnomalyDetection
Identifies anomalous pixels in hyperspectral images.

**Key Methods:**
- `rx_detector()`: RX (Mahalanobis Distance) anomaly detector
- `sam_detection()`: Spectral Angle Mapper anomaly detection
- `isolation_forest()`: Unsupervised isolation forest method

**Example:**
```python
anomaly = AnomalyDetection()
rx_scores = anomaly.rx_detector(image)
sam_scores = anomaly.sam_detection(image, reference_spectrum)
```

### 7. HyperspectralUtils
Utility functions for common operations.

**Key Methods:**
- `normalize_image()`: Min-max normalization
- `calculate_statistics()`: Compute band statistics
- `create_rgb_composite()`: Create false-color composites
- `export_geotiff()`: Save with geospatial metadata

**Example:**
```python
utils = HyperspectralUtils()
rgb = utils.create_rgb_composite(image, bands=[29, 19, 9])
utils.export_geotiff(processed_image, output_path, crs, transform)
```

## Requirements

### Python Packages
- numpy >= 1.19.0
- pandas >= 1.1.0
- matplotlib >= 3.3.0
- seaborn >= 0.11.0
- scipy >= 1.5.0
- rasterio >= 1.1.0
- scikit-learn >= 0.23.0

### Optional Dependencies
- GDAL >= 3.0 (for advanced geospatial operations)
- PyTorch (for deep learning-based anomaly detection)

### System Requirements
- 8GB RAM (minimum for typical AVIRIS scenes)
- 16GB RAM (recommended for large hyperspectral datasets)
- 50GB disk space for sample datasets

## Installation

1. **Clone or download the repository:**
```bash
cd Hyperspectral_P
```

2. **Create a Python virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages:**
```bash
pip install numpy pandas matplotlib seaborn scipy rasterio scikit-learn
```

## Usage

### Running the Notebook

1. **Open the Jupyter notebook:**
```bash
jupyter notebook data_preprocessing.ipynb
```

2. **Run cells sequentially:**
   - Execute cell 1: Import libraries and set up directories
   - Execute cell 2-8: Initialize processing modules
   - Execute cell 9+: Apply processing pipeline steps

### Basic Workflow

```python
# 1. Load and validate data
acq = DataAcquisition()
metadata = acq.validate_metadata('data/raw/image.tif')

# 2. Radiometric correction
rad_corrector = RadiometricCorrection(gain, offset)
radiance = rad_corrector.dn_to_radiance(dn_image)

# 3. Atmospheric correction
atm_corrector = AtmosphericCorrection()
surface_ref = atm_corrector.quac_simple(radiance)

# 4. Dimensionality reduction
dimred = DimensionalityReduction()
pca_result, variance = dimred.compute_pca(surface_ref, n_components=10)

# 5. Spectral unmixing
unmixing = SpectralUnmixing()
endmembers = unmixing.extract_endmembers_ppi(surface_ref, n_endmembers=5)
abundances = unmixing.linear_spectral_unmixing(surface_ref, endmembers)

# 6. Anomaly detection
anomaly = AnomalyDetection()
rx_scores = anomaly.rx_detector(surface_ref)

# 7. Export results
utils = HyperspectralUtils()
utils.export_geotiff(abundances[:,:,0], 'outputs/abundance_1.tif', crs, transform)
```

## Algorithms and Methods

### Radiometric Correction
- **DN to Radiance**: $L = M_L \cdot DN + A_L$
- **Thermal Brightness Temperature**: $T = \frac{K_2}{\log(K_1/L + 1)}$

### Atmospheric Correction
- **QUAC**: Dark object subtraction
- **FLAASH**: Empirical transmittance model

### Dimensionality Reduction
- **MNF**: Noise-whitened eigenvalue decomposition
- **PCA**: Standard principal component analysis
- **Band Selection**: Entropy, variance, and correlation metrics

### Spectral Unmixing
- **Linear Unmixing**: $X = M \cdot A + N$ (Non-negative Least Squares)
- **PPI**: Iterative random projections for endmember extraction

### Anomaly Detection
- **RX Detector**: Mahalanobis distance from background distribution
- **SAM**: Spectral angle between target and reference

## Output Directories

### `/data`
- **raw/**: Original hyperspectral imagery
- **processed/**: All processed outputs organized by correction type
- **statistics/**: Band-wise min, max, mean, std, histograms

### `/outputs`
- **unmixing_results/**: Abundance maps for each endmember
- **anomaly_maps/**: RX and SAM anomaly score maps
- **statistics/**: JSON files with numerical analysis results

### `/figures`
- **rgb_compositions/**: False-color RGB visualizations
- **band_statistics/**: Histograms and statistical plots
- **spectral_analysis/**: Spectral profile comparisons

## Performance Metrics

- **Processing Time**: ~5-15 mins for typical 100 MB hyperspectral image
- **Memory Usage**: ~3-5x input file size for intermediate calculations
- **Accuracy**: SAM angle error < 5° for pure endmembers

## Troubleshooting

### Memory Issues
If you encounter out-of-memory errors:
- Reduce image dimensions using band selection
- Process smaller image tiles
- Increase swap memory or upgrade RAM

### Geometric Registration Errors
- Verify coordinate reference system (CRS) consistency
- Check for datum mismatches
- Ensure GCPs (Ground Control Points) are accurate

### Spectral Unmixing Convergence
- Verify endmember spectral variability (eigenvalues > 1)
- Increase iterations for PPI extraction
- Use spectral angle constraints

## References

### Key Publications
- Benediktsson, J.A., et al. (2011). "Feature Extraction for Multispectral and Hyperspectral Image Analysis." Proceedings of the IEEE.
- Chang, C.I. (2003). "Hyperspectral Imaging: Techniques for Spectral Detection and Classification."
- Lee, C., & Landgrebe, D.A. (1993). "Analyzing High-Dimensional Multispectral Data."

### Useful Resources
- [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
- [Rasterio Documentation](https://rasterio.readthedocs.io/)
- [Scikit-Learn Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html)

## Contributing

To extend this pipeline:

1. Add new processing modules as classes
2. Follow the existing pattern of `__init__` and static/instance methods
3. Document with docstrings including Args, Returns, and Examples
4. Test with sample data before committing

### Areas for Enhancement
- GPU acceleration for large-scale processing
- Deep learning-based anomaly detection
- Advanced FLAASH implementation with full physics model
- Kriging-based spatial interpolation
- Temporal change detection for time series

## License

This project is provided as-is for research and educational purposes.

## Contact & Support

For questions or issues:
- Review the notebook comments and docstrings
- Check the References section for algorithm details
- Verify data format compatibility with supported sensors

## Version History

- **v1.0** (Current): Initial comprehensive hyperspectral preprocessing pipeline
  - Core modules for radiometric and atmospheric correction
  - MNF and PCA implementations
  - Linear spectral unmixing with PPI
  - RX and SAM anomaly detection
  - Comprehensive visualization tools

## Acknowledgments

This implementation incorporates algorithms and methods from the hyperspectral remote sensing community. Special thanks to the developers of rasterio, NumPy, SciPy, and scikit-learn for providing essential tools.
