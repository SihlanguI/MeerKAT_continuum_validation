#!/usr/bin/env python
import argparse
import logging
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy import wcs
from astropy import coordinates


def read_fits(pathfits):
    '''
    This functions reads in the fits file
    Input: Path to the fits file
    Output: Fits file object
    '''
    fl = fits.open(pathfits)
    images = fl[0]
    return images


def get_positions(path):
    '''
    This function reads the fits file and returns the sky coordinate
    of the phase centre.
    Input: path to the fits file
    Output: Sky coordinate of the phase centre
    '''
    images = read_fits(path)
    # Get pointing centre of observation
    phase_centre_ra = images.header['OBSRA']
    phase_centre_dec = images.header['OBSDEC']
    # Convert to astropy.coordinates.SkyCoord object
    phase_center = coordinates.SkyCoord(phase_centre_ra, phase_centre_dec, unit=(u.deg, u.deg))
    # Get positions of other sources
    source_pos = wcs.WCS(images.header)
    return phase_center, source_pos


def radial_offset(phase_center, source_pos, row, col):
    '''
    This function gets the position in ra and dec of a pixel and determine its
    radial offset from phase centre.
    Input: phase center and source positions
    output: A separation vector between phase centre and source positions in radians.
    '''
    p2w = source_pos.pixel_to_world(row, col, 0, 0)[0]
    separation_rad = p2w.separation(phase_center).rad
    return separation_rad


def central_freq(path):
    '''
    Function to get central frequency of each frequency plane.
    Input : image header
    Output : List of central frequency in MHz of each frequency plane.
    '''
    images = read_fits(path)
    c_freq_plane = []
    for i in range(1, images.header['NSPEC']+1):
        c_freq_plane.append(images.header['FREQ{0:04}'.format(i)])
    return np.array(c_freq_plane)


def beam_pattern(separation_rad, path, c_freq=None):
    '''
    Making beam pattern using the DEEP2 paper
    Input : Radial seperation array and path to the fits file
    Output : Beam pattern
    '''
    if c_freq:
        rho = separation_rad
        c = c_freq*1e6
        # Degrees to radians conversion
        conv = np.pi / 180.
        # converting arcminutes to radians
        v_beam_rad = 89.5 / 60. * conv
        h_beam_rad = 86.2 / 60. * conv
        # Taking the Geometric mean for the veritical and horizontal cut through the beam.
        vh_beam_mean = np.sqrt(v_beam_rad*h_beam_rad)
        flux_density = []
        nu = c/1.e9
        theta_b = vh_beam_mean*(1./nu)
        ratio = rho/theta_b
        num = np.cos(1.189*np.pi*ratio)
        dem = 1-4*(1.189*ratio)**2
        a_b = (num/dem)**2
        flux_density.append(a_b)
    else:
        rho = separation_rad
        c = central_freq(path)
        # Degrees to radians conversion
        conv = np.pi / 180.
        # converting arcminutes to radians
        v_beam_rad = 89.5 / 60. * conv
        h_beam_rad = 86.2 / 60. * conv
        # Taking the Geometric mean for the veritical and horizontal cut through the beam.
        vh_beam_mean = np.sqrt(v_beam_rad*h_beam_rad)
        flux_density = []
        for nu in c:
            nu = nu/1.e9
            theta_b = vh_beam_mean*(1./nu)
            ratio = rho/theta_b
            num = np.cos(1.189*np.pi*ratio)
            dem = 1-4*(1.189*ratio)**2
            a_b = (num/dem)**2
            flux_density.append(a_b)
    return flux_density


def mad(data):
    '''
    Calculating meadian absolute deviation (MAD).
    '''
    MAD_TO_SD = 1.4826
    med = np.nanmedian(data)
    dev = np.abs(data - med)
    return med, MAD_TO_SD * np.nanmedian(dev)


def standard_deviation(data):
    '''
    Calcalating the standard deviation of each frequency plane using MAD.
    Returns: Weights per each frequency plane
    '''
    diff = 1
    i = 0
    data = data[data != 0.0]
    if len(data) == 0:
        return float(0.0), float(0.0)
    med, sd = mad(data)
    while diff > 0:
        i += 1
        if i > 50:
            return sd
        old_sd = sd
        old_len = len(data)
        cut = np.abs(data - med) < 5.0 * sd
        if np.all(~cut):
            return sd
        data = data[cut]
        sd = mad(data)[1]
        diff = old_len - len(data)
        if sd == 0.0:
            return old_sd
    return 1/(sd)**2


def weigted_average(arr, weights):
    '''
    Computing weighted average of all the frequency planes.
    '''
    wt_average = np.average(arr, weights=weights, axis=0)
    return wt_average


def primary_beam_correction(beam_pattern, path):
    '''
    Correcting for the primary beam effects.
    '''
    raw_image = read_fits(path)
    raw_img = []
    try:
        nterm = raw_image.header['NTERM']
    except KeyError:
        nterm = _get_value_from_history('NTERM', raw_image.header)
    nterm = int(nterm)
    for j in range(nterm, raw_image.header['NSPEC']+nterm):
        raw_img.append(np.ravel(raw_image.data[0, j, :, :]))
    snr = []
    pbc_image = []
    for i in range(len((beam_pattern))):
        # Getting all the values with attenuated flux of less than 10% of the peak
        ind = np.argwhere(beam_pattern[i] <= 0.10)
        if len(ind) > 0:
            beam_pattern[i][ind] = np.nan
        # Getting the rms noise in each image (before primary beam correction)
        snr.append(standard_deviation(raw_img[i]))
        # To correct the effect of the beam we divide by the beam.
        ratio = raw_img[i] / beam_pattern[i]
        pbc_image.append(ratio)
    # Primary beam corrected (pbc) image
    pbc_image = np.array(pbc_image)
    snr = np.array(snr)
    # Calculating a weighted average of the indidvidual frequency lane images
    corr_image = weigted_average(pbc_image, snr)
    corr_image = corr_image.reshape(raw_image.data.shape[2], raw_image.data.shape[3])
    # Add new axis
    corr_image = corr_image[np.newaxis, np.newaxis, :, :]
    return corr_image


def _get_value_from_history(keyword, header):
    ''''
    Return the value of a keyword from the FITS HISTORY in header
    Assumes keyword is found in a line of the HISTORY with format: 'keyword = value'
    Input: Keyword [e.g BMAJ, ClEANBMJ, BMIN] and Image header
    '''
    for history in header['HISTORY']:
        line = history.replace('=', ' ').split()
        try:
            ind = line.index(keyword)
        except ValueError:
            continue
        return line[ind + 1]
    raise KeyError(f'{keyword} not found in HISTORY') 
    

def write_new_fits(pbc_image, path, outputFilename):
    '''
    Write out a new fits image with primary beam corrected continuum in its first  plane
    '''
    images = read_fits(path)
    hdr = images.header
    newhdr = hdr.copy()
    # change the frequency plane keywords, we don't want multiple frequency axes
    newhdr['CTYPE3'] = 'FREQ'
    newhdr['NAXIS3'] = 1
    newhdr['CDELT3'] = 1.0
    try:
        if newhdr['CLEANBMJ'] > 0:
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
    data_new = pbc_image[:, 0:1, :, :]  # the shape of data_new = (1,1, npix, npix)
    new_hdu = fits.PrimaryHDU(header=newhdr, data=data_new)
    new_hdu.writeto(outputFilename, overwrite=True)
    return


def intialize_logs():
    '''
    Initializing the log settings
    '''
    logging.basicConfig(format='%(message)s', level=logging.INFO)


def create_parser():
    parser = argparse.ArgumentParser("Input an MeerKAT SDP pipeline continuum image and prouduce "
                                     "primary beam corrected image in a direcectory same as that "
                                     "of an input image.")
    parser.add_argument('input',
                               help='MeerKAT continuum uncorrected primary beam fits file')
    return parser


def main():
    # Initializing the log settings
    intialize_logs()
    logging.info("MeerKAT SDP continuum image primary beam correction.")
    parser = create_parser()
    args = parser.parse_args()
    path = args.input
    logging.info('----------------------------------------')
    logging.info('Reading in the fits file')
    data = read_fits(path)
    logging.info('----------------------------------------')
    logging.info('Getting the position of the phase centre')
    phase_center, source_pos = get_positions(path)
    # Getting pixel indice
    logging.info('----------------------------------------')
    logging.info('Getting the indices of the pixels')
    row, col = np.indices((data.shape[2], data.shape[3]))
    row = np.ravel(row)
    col = np.ravel(col)
    # Getting radial separation beween sorces and the phase centre
    logging.info('----------------------------------------')
    logging.info('Getting the radial separation between sources and the phase centre')
    separation_rad = radial_offset(phase_center, source_pos, row, col)
    logging.info('----------------------------------------')
    logging.info('Getting the beam pattern for each frequecy plane based on the DEEP2 paper.')
    bp = beam_pattern(separation_rad, path)
    # pbc - primary beam corrected
    logging.info('----------------------------------------')
    logging.info('Doing the primary beam correction in each frequency plane and averaging')
    pbc_image = primary_beam_correction(bp, path)
    # Saving the primary beam corrected image
    logging.info('----------------------------------------')
    logging.info('Saving the primary beam corrected image')
    ind = [i for i in range(len(path)) if path[i] == '.' and (path[i+1:i+5] == 'fits' or
                                                              path[i+1:i+5] == 'FITS')] 
    outputpath = (path[0:ind[0]] + '_PB.fits')
    write_new_fits(pbc_image, path, outputFilename=outputpath)
    logging.info('------------------DONE-------------------')


if __name__ == "__main__":
    main()
