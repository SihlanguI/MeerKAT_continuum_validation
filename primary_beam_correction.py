#!/usr/bin/env python
import argparse
import logging
import os
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy import wcs
from astropy import coordinates


def read_fits(pathfits):
    """
    This functions reads in the fits file
    Input: Path to the fits file
    Output: Fits file object
    """
    fl = fits.open(pathfits)
    images = fl[0]
    return images


def get_positions(path):
    """
    This function reads the fits file and returns the sky coordinate
    of the phase centre.
    Input: path to the fits file
    Output: Sky coordinate of the phase centre
    """
    image_wcs = wcs.WCS(path)
    # Get pointing centre of observation
    phase_centre_ra = image_wcs.celestial.wcs.crval[0]
    phase_centre_dec = image_wcs.celestial.wcs.crval[1]
    # Convert to astropy.coordinates.SkyCoord object
    phase_center = coordinates.SkyCoord(phase_centre_ra, phase_centre_dec, unit=(u.deg, u.deg))
    return phase_center, image_wcs


def radial_offset(phase_center, image_wcs, row, col):
    """
    This function gets the position in ra and dec of a pixel and determine its
    radial offset from phase centre.
    Input: phase center and wcs image
    output: A separation vector between phase centre and source positions in radians.
    """
    p2w = image_wcs.pixel_to_world(row, col, 0, 0)[0]
    separation_rad = p2w.separation(phase_center).rad
    return separation_rad


def central_freq(path):
    """
    Function to get central frequency of each frequency plane.
    Input : image header
    Output : List of central frequency in MHz of each frequency plane.
    """
    images = read_fits(path)
    c_freq_plane = []
    for i in range(1, images.header['NSPEC']+1):
        c_freq_plane.append(images.header['FREQ{0:04}'.format(i)])
    return np.array(c_freq_plane)


def cosine_power_pattern(separation_rad, c_freq):
    """
    Power patterns for a given frequency based on the Cosine-squared power approximation
    from Mauch et al. (2020).
    Input: Radial separation array and a central frequency
    """
    rho = separation_rad
    # Degrees to radians conversion
    conv = np.pi / 180.
    # converting arcminutes to radians
    v_beam_rad = 89.5 / 60. * conv
    h_beam_rad = 86.2 / 60. * conv
    # Taking the Geometric mean for the veritical and horizontal cut through the beam.
    vh_beam_mean = np.sqrt(v_beam_rad*h_beam_rad)
    flux_density = []
    for nu in c_freq:
        nu = nu/1.e9
        theta_b = vh_beam_mean*(1./nu)
        ratio = rho/theta_b
        num = np.cos(1.189*np.pi*ratio)
        dem = 1-4*(1.189*ratio)**2
        a_b = (num/dem)**2
        flux_density.append(a_b)
    return flux_density


def beam_pattern(path):
    """
    Making beam pattern using the Cosine-squared power approximation from Mauch et al. (2020)
    """
    # Reading the fits file
    data = read_fits(path)
    # Getting pixel indices
    row, col = np.indices((data.shape[2], data.shape[3]))
    row = np.ravel(row)
    col = np.ravel(col)
    # Getting the central frequency of the image header given.
    c_freq = central_freq(path)
    # Getting radial separation between sources and the phase centre
    phase_center, image_wcs = get_positions(path)
    # Calling the radial_offset function
    separation_rad = radial_offset(phase_center, image_wcs, row, col)
    # Calling the cosine_power_pattern function
    beam_list = cosine_power_pattern(separation_rad, c_freq)
    return beam_list


def mad(data):
    """
    Calculating median absolute deviation (MAD).
    """
    MAD_TO_SD = 1.4826
    med = np.nanmedian(data)
    dev = np.abs(data - med)
    return med, MAD_TO_SD * np.nanmedian(dev)


def standard_deviation(data):
    """
    Calcalating the standard deviation of each frequency plane using MAD. Rejecting pixels more
    than 5 sigma from the mean until either no more pixels are rejected or a maximum  of 50
    iterations is reached.
    Returns: Weights per each frequency plane
    """
    diff = 1
    i = 0
    data = data[data != 0.0]
    if len(data) == 0:
        return float(0.0), float(0.0)
    med, sd = mad(data)
    while diff > 0:
        i += 1
        if i > 50:
            return 1/(sd)**2
        old_sd = sd
        old_len = len(data)
        cut = np.abs(data - med) < 5.0 * sd
        if np.all(~cut):
            return 1/(sd)**2
        data = data[cut]
        med, sd = mad(data)
        diff = old_len - len(data)
        if sd == 0.0:
            return 1/(old_sd)**2
    return 1/(sd)**2


def weighted_average(arr, weights, mask_image):
    """
    Computing weighted average of all the frequency planes.
    """
    masked_arr = np.ma.MaskedArray(arr, mask=mask_image)
    wt_average = np.average(masked_arr, weights=weights, axis=0)
    return wt_average.data


def primary_beam_correction(beam_pattern, path):
    """
    Correcting for the primary beam effects.
    """
    raw_image = read_fits(path)
    nterm = raw_image.header['NTERM']
    snr = []
    pbc_image = []
    mask_image = []
    # Getting all the values with attenuated flux of less than 10% of the peak
    beam_mask = beam_pattern[-1] <= 0.1
    for i in range(len((beam_pattern))):
        beam_pattern[i][beam_mask] = np.nan
        # Getting the rms noise in each image (before primary beam correction)
        snr.append(standard_deviation(np.ravel(raw_image.data[0, i+nterm, :, :])))
        # To correct the effect of the beam we divide by the beam.
        ratio = np.ravel(raw_image.data[0, i + nterm, :, :]) / beam_pattern[i]
        mask = np.isnan(ratio)
        mask_image.append(mask)
        pbc_image.append(ratio)
    # Primary beam corrected (pbc) image
    pbc_image = np.array(pbc_image)
    mask_image = np.array(mask_image)
    snr = np.array(snr)
    # Calculating a weighted average of the individual frequency plane images
    corr_image = weighted_average(pbc_image, snr, mask_image)
    # Add new axis
    corr_image = corr_image.reshape(1, 1, raw_image.data.shape[2], raw_image.data.shape[3])
    return corr_image


def _get_value_from_history(keyword, header):
    """
    Return the value of a keyword from the FITS HISTORY in header
    Assumes keyword is found in a line of the HISTORY with format: 'keyword = value'
    Input: Keyword [e.g BMAJ, CLEANBMJ, BMIN] and Image header
    """
    for history in header['HISTORY']:
        line = history.replace('=', ' ').split()
        try:
            ind = line.index(keyword)
        except ValueError:
            continue
        return line[ind + 1]
    raise KeyError(f'{keyword} not found in HISTORY')
    return


def write_new_fits(pbc_image, path, outputFilename):
    """
    Write out a new fits image with primary beam corrected continuum in its first  plane
    """
    images = read_fits(path)
    hdr = images.header
    newhdr = hdr.copy()
    # change the frequency plane keywords, we don't want multiple frequency axes
    newhdr['CTYPE3'] = 'FREQ'
    newhdr['NAXIS3'] = 1
    newhdr['CDELT3'] = 1.0
    try:
        if 'CLEANBMJ' in newhdr and newhdr['CLEANBMJ'] > 0:
            # add in required beam keywords
            newhdr['BMAJ'] = newhdr['CLEANBMJ']
            newhdr['BMIN'] = newhdr['CLEANBMN']
            newhdr['BPA'] = newhdr['CLEANBPA']
        else:
            # Checking CLEANBMAJ in the history
            newhdr['BMAJ'] = float(_get_value_from_history('BMAJ', newhdr))
            newhdr['BMIN'] = float(_get_value_from_history('BMIN', newhdr))
            newhdr['BPA'] = float(_get_value_from_history('BPA', newhdr))
    except KeyError:
        logging.error('Exception occurred, keywords not found', exc_info=True)
    new_hdu = fits.PrimaryHDU(header=newhdr, data=pbc_image)
    new_hdu.writeto(outputFilename, overwrite=True)
    return


def intialize_logs():
    """
    Initializing the log settings
    """
    logging.basicConfig(format='%(message)s', level=logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser("Input a MeerKAT SDP pipeline continuum image and produce "
                                     "primary beam corrected image in a direcectory same as that "
                                     "of an input image.")
    parser.add_argument('input',
                        help='MeerKAT continuum uncorrected primary beam fits file')
    return parser


def main():
    # Initializing the log settings
    intialize_logs()
    logging.info('MeerKAT SDP continuum image primary beam correction.')
    parser = create_parser()
    args = parser.parse_args()
    path = os.path.abspath(args.input)
    logging.info('----------------------------------------')
    logging.info('Getting the beam pattern for each frequecy plane based on the '
                 'Cosine-squared power approximation from Mauch et al. (2020).')
    bp = beam_pattern(path)
    # pbc - primary beam corrected
    logging.info('----------------------------------------')
    logging.info('Doing the primary beam correction in each frequency plane and averaging')
    pbc_image = primary_beam_correction(bp, path)
    # Saving the primary beam corrected image
    logging.info('----------------------------------------')
    logging.info('Saving the primary beam corrected image')
    head, tail = os.path.split(path)
    fname, fext = tail.split('.')
    outputpath = (head+fname+'_PB'+fext)
    write_new_fits(pbc_image, path, outputFilename=outputpath)
    logging.info('------------------DONE-------------------')


if __name__ == "__main__":
    main()
