import numpy as np
import pyspedas, pytplot, re, os, cdflib, bisect, wget
from datetime import date, timedelta, datetime

class span:
    def __init__(self, tstart, tend):
        # formatting to datetime
        self.tstart = self.parse_to_datetime(tstart)
        self.tend = self.parse_to_datetime(tend)

        # getting an array of the different dates in the range
        self.day_arr = np.array([])
        self.get_dates()
        self.Ndays = len(self.day_arr)

        # getting start and end times for each day
        self.datetstart_flag = None
        self.datetend_flag = None
        self.get_timeflag_for_dates()

        # the VDF dictionary which gets updated for each day (to avoid loading too much into memory)
        self.tidx_start, self.tidx_end, self.VDF_dict = None, None, {}
    
    def parse_to_datetime(self, tstring):
        year, month, date, hour, minute, second = np.asarray(re.split('[-,/,:]', tstring)).astype('int')
        return datetime(year, month, date, hour, minute, second)
    
    def get_timeflag_for_dates(self):
        self.datetstart_flag = np.ones(len(self.day_arr)) * -1
        self.datetend_flag = np.ones(len(self.day_arr)) * -1

        self.datetstart_flag[0] = 1
        self.datetend_flag[-1] = 1

    def get_dates(self):
        delta = self.tend - self.tstart

        for i in range(delta.days + 1):
            self.day_arr = np.append(self.day_arr, self.tstart + timedelta(days=i))

    def get_VDFdict_at_day(self, cdf_VDfile):
        epochSlice  = cdf_VDfile['EPOCH'][self.tidx_start: self.tidx_end]
        thetaSlice  = cdf_VDfile['THETA'][self.tidx_start: self.tidx_end,:]
        phiSlice    = cdf_VDfile['PHI'][self.tidx_start: self.tidx_end,:]
        energySlice = cdf_VDfile['ENERGY'][self.tidx_start: self.tidx_end,:]
        efluxSlice  = cdf_VDfile['EFLUX'][self.tidx_start: self.tidx_end,:]

        mass_p = 0.010438870      #proton mass in units eV/c^2 where c = 299792 km/s
        charge_p = 1              #proton charge in units eV

        # Define VDF
        numberFluxSlice = efluxSlice/energySlice
        vdfSlice = numberFluxSlice*(mass_p**2)/((2E-5)*energySlice)

        # making the dictionary
        vdf_bundle = {}

        vdf_bundle['epoch'] = epochSlice
        vdf_bundle['vdf'] = vdfSlice[:,:,::-1,:]
        vdf_bundle['theta'] = thetaSlice[:,:,::-1,:]
        vdf_bundle['phi'] = phiSlice[:,:,::-1,:]
        vdf_bundle['energy'] = energySlice[:,:,::-1,:]

        return vdf_bundle

    def get_L3_monents_at_t(L3_data, tSliceIndex):
        # making the dictionary
        l3_data_bundle = {}

        # extracting out the required L3 data
        l3_data_bundle['MAGF_INST'] = L3_data['MAGF_INST'][tSliceIndex,:]
        l3_data_bundle['VEL_INST'] = L3_data['VEL_INST'][tSliceIndex,:]
        l3_data_bundle['DENS'] = L3_data['DENS'][tSliceIndex]

        return l3_data_bundle

    def start_new_day(self, day_idx):
        self.tidx_start, self.tidx_end, self.VDF_dict = None, None, {}

        # load the data from CDF file for this day (download file if needed)
        fullday_dat = self.download_VDF_file(self.day_arr[day_idx])

        # getting the limits in time index for this day
        epoch = cdflib.cdfepoch.to_datetime(fullday_dat['EPOCH'])
        if(self.datetstart_flag[day_idx] == 1):
            self.tidx_start = bisect.bisect_left(epoch, np.datetime64(self.tstart))
        else: 
            self.tidx_start = 0
        if(self.datetend_flag[day_idx] == 1):
            self.tidx_end = bisect.bisect_left(epoch, np.datetime64(self.tend))
        else: 
            self.tidx_end = len(epoch)

        # clipping the VDF data in time for this day
        self.VDF_dict = self.get_VDFdict_at_day(fullday_dat)

        # clipping the L3 spi data
        self.L3_data_fullday = self.download_L3_data(self.day_arr[day_idx])

    def VDfile_directoryRemote(self, user_datetime):
        '''
        Function to download a new CDF file if not found in the data directory.
        '''
        def yyyymmdd(dt) : return f"{dt.year:04d}{dt.month:02d}{dt.day:02d}"

        VDF_RemoteDir = f'http://w3sweap.cfa.harvard.edu/pub/data/sci/sweap/spi/L2/spi_sf00/{user_datetime.year:04d}/{user_datetime.month:02d}/'
        VDF_filename = f'psp_swp_spi_sf00_L2_8Dx32Ex8A_{yyyymmdd(user_datetime)}_v04.cdf'

        print(VDF_RemoteDir, VDF_filename)

        return VDF_RemoteDir, VDF_filename

    def download_VDF_file_old(self, user_datetime):
        # Import from file directory
        VDF_RemoteDir, VDF_filename = self.VDfile_directoryRemote(user_datetime)

        #check if file is already downloaded. If so, skip download. If not, download in local directory.
        if os.path.isfile(f'./data/{VDF_filename}'):
            print(f"File already exists in local directory - [./data/{VDF_filename}]")
            VDfile = f'./data/{VDF_filename}'
        else:
            print("File doesn't exist. Downloading ...")
            VDfile = wget.download(VDF_RemoteDir + VDF_filename, out='./data')

        # open CDF file
        dat_raw = cdflib.CDF(VDfile)
        dat = {}

        # creating the data slice (1 day max)
        dat['EPOCH']  = dat_raw['EPOCH']
        dat['THETA']  = dat_raw['THETA'].reshape((-1,8,32,8))
        dat['PHI']    = dat_raw['PHI'].reshape((-1,8,32,8))
        dat['ENERGY'] = dat_raw['ENERGY'].reshape((-1,8,32,8))
        dat['EFLUX']  = dat_raw['EFLUX'].reshape((-1,8,32,8))

        return dat

    def download_VDF_file(self, user_datetime, CREDENTIALS=None):
        if CREDENTIALS:
            files = pyspedas.psp.spi(trange, datatype='spi_sf00', level='L2', notplot=True, time_clip=True, downloadonly=True, last_version=True, username=CREDENTIALS[0], password=CREDENTIALS[1])
        else:
            files = pyspedas.psp.spi(trange, datatype='spi_sf00_8dx32ex8a', level='l2', notplot=True, time_clip=True, downloadonly=True, last_version=True)

        dat_raw = cdflib.cdf_to_xarray(*files)
        dat = {}

        # creating the data slice (1 day max)
        dat['EPOCH']  = dat_raw['Epoch'].data
        dat['THETA']  = dat_raw['THETA'].data.reshape((-1,8,32,8))
        dat['PHI']    = dat_raw['PHI'].data.reshape((-1,8,32,8))
        dat['ENERGY'] = dat_raw['ENERGY'].data.reshape((-1,8,32,8))
        dat['EFLUX']  = dat_raw['EFLUX'].data.reshape((-1,8,32,8))

        return dat


    def download_L3_data(self, user_datetime):
        yyyy, mm, dd = user_datetime.year, user_datetime.month, user_datetime.day

        trange = [f'{yyyy}-{mm}-{dd}/00:00:00', f'{yyyy}-{mm}-{dd}/23:59:59']
        try:
            spi_vars = pyspedas.psp.spi(trange=trange, datatype='spi_sf00_l3_mom', level='l3',
                                        time_clip=True, get_support_data= True, varnames=['*'],
                                        notplot=True, downloadonly=True)
            dat = cdflib.CDF(spi_vars[0])
        except:
            spi_vars = pyspedas.psp.spi(trange=trange, datatype='spi_sf00', level='L3',
                                        time_clip=True, get_support_data= True, varnames=['*'],
                                        notplot=True, downloadonly=True, username='sbdas',
                                        password='SlapHappeeGranpappy01238')
            dat = cdflib.CDF(spi_vars[0])

        return dat

    def get_L3_monents_at_t(self, tSliceIndex):
        # making the dictionary
        l3_data_bundle = {}

        # extracting out the required L3 data
        l3_data_bundle['MAGF_INST'] = self.L3_data_fullday['MAGF_INST'][tSliceIndex,:]
        l3_data_bundle['VEL_INST'] = self.L3_data_fullday['VEL_INST'][tSliceIndex,:]
        l3_data_bundle['DENS'] = self.L3_data_fullday['DENS'][tSliceIndex]

        return l3_data_bundle