# -*- coding: utf-8 -*-
import json, os, sys, inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import astropy.units as u
import astropy.constants as const
from astropy.io import ascii
import scipy.interpolate as interpolate

class EtaBase(object):
    '''Occurrence rate object which takes a json script pointing to occurrence
    rate files and fills the relevant attributes
    
    Args:
        scriptfile (string): path to json script file
        
    Attributes:
        
    The stellar data is from:
    "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    Eric Mamajek (JPL/Caltech, University of Rochester) 
    Version 2017.10.19
    
    Forecaster data is from Chen and Kipping 2016
        
    '''
    
    def __init__(self, scriptfile=None, name=None, title=None, suffix=None, **specs):
        # get specs from json script file
        if scriptfile is not None:
            assert os.path.isfile(scriptfile), "%s is not a file." % scriptfile
            try:
                script = open(scriptfile).read()
                specs_from_file = json.loads(script)
                specs_from_file.update(specs)
            except ValueError as err:
                print("Error: %s: Input file `%s' improperly formatted."%(self._modtype,
                        scriptfile))
                print("Error: JSON error was: %s"%err)
                # re-raise here to suppress the rest of the backtrace.
                # it is only confusing details about the bowels of json.loads()
                raise ValueError(err)
            except:
                print("Error: %s: %s"%(self._modtype, sys.exc_info()[0]))
                raise
        else:
            specs_from_file = {}
        specs.update(specs_from_file)
        
        # initialize attributes
        self.RpData = {}
        self.MsiniData = {}
        self.PData = {}
        self.aData = {}
        self.starData = {}
        self.Etas = {}
        self.fname = {}
        self.specs = specs
        
        # fill in RpData, MsiniData, PData, aData from script file
        keys1 = ['input','unit','scale']
        for key in keys1:
            self.RpData[key] = self.specs['RpData'][key]
            self.MsiniData[key] = self.specs['MsiniData'][key]
            self.PData[key] = self.specs['PData'][key]
            self.aData[key] = self.specs['aData'][key]
        self.RpData['range'] = np.array(self.specs['RpData']['range'])
        self.MsiniData['range'] = np.array(self.specs['MsiniData']['range'])
        self.PData['range'] = np.array(self.specs['PData']['range'])
        self.aData['range'] = np.array(self.specs['aData']['range'])
        
        # fill in starData from script file
        keys2 = ['type','Teff']
        for key in keys2:
            self.starData[key] = self.specs['starData'][key]
        
        # fill in name and file information of data
        keys3 = ['name','title','suffix']
        for key in keys3:
            self.fname[key] = self.specs['fname'][key]
        
        # fill in Etas from file
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        datafolderpath = os.path.join(classpath, self.fname['name'])
        # filename for etas
        fEta = os.path.join(datafolderpath,'eta'+self.fname['suffix'])
        fsigp = os.path.join(datafolderpath,'sigma_p'+self.fname['suffix'])
        fsign = os.path.join(datafolderpath,'sigma_n'+self.fname['suffix'])
        if not os.path.isfile(fEta):
            raise ValueError('File %s could not be found.' % fEta)
        if not os.path.isfile(fsigp):
            fsigp = None
        if not os.path.isfile(fsign):
            fsign = None
        
        # pull in data from supplied files
        print('Pulling occurrence rate data from {}'.format(self.fname['name']))
        ntypes = len(self.starData['type'])
        for i in xrange(ntypes):
            self.Etas[self.starData['type'][i]] = {}
        self.Etas = self.readfiles(fEta,'eta')
        # pull in positive sigma 
        if fsigp is not None:
            for i in xrange(ntypes):
                tmp_dict = self.readfiles(fsigp,'sigp')
                self.Etas[self.starData['type'][i]].update(tmp_dict[self.starData['type'][i]])
            print('Positive sigma data found and loaded')
        
        # pull in negative sigma
        if fsign is not None:
            for i in xrange(ntypes):
                tmp_dict = self.readfiles(fsign, 'sign')
                self.Etas[self.starData['type'][i]].update(tmp_dict[self.starData['type'][i]])
            print('Negative sigma data found and loaded')
        
        # get necessary stellar data and interpolants
        starpath = os.path.join(classpath,'EEM_dwarf_UBVIJHK_colors_Teff.txt')
        data = ascii.read(starpath,fill_values=[('...',np.nan),('....',np.nan),('.....',np.nan)])
        Teff = np.array([])
        Msun = np.array([])
        for T in data['Teff'].data:
            Teff = np.hstack((Teff,T))
        for M in data['Msun'].data:
            if type(M) == np.ma.core.MaskedConstant:
                Msun = np.hstack((Msun,np.nan))
            else:
                Msun = np.hstack((Msun,M))
        vals = np.where(~np.isnan(Msun))
        Ts = Teff[vals]
        Ms = Msun[vals]
        self.Msun_from_Teff = interpolate.InterpolatedUnivariateSpline(Ts[::-1],Ms[::-1],ext=1)

        # convert given data to missing data types
        if self.PData['input']:
            self.P_to_a()
        if self.aData['input']:
            self.a_to_P()
        if self.RpData['input']:
            self.Rp_to_M()
        if self.MsiniData['input']:
            self.M_to_Rp()
        
    def readfiles(self, fname, key):
        '''Reads in eta and sigma files and stores them in self.Etas'''
        ntypes = len(self.starData['type'])
        tmp = np.genfromtxt(fname, comments='%')
        nlines, nPa = tmp.shape
        nRM = nlines/ntypes
        tmp_val = tmp.reshape((ntypes,nRM,nPa))
        tmp_val[tmp_val==0.] = np.nan # zero values interpreted as nan
        m = {}
        for i in xrange(ntypes):
            tmp_dict = {key: tmp_val[i,:,:]}
            m[self.starData['type'][i]] = tmp_dict
        
        return m
    
    def get_a_from_P(self,P,Ms):
        '''Converts list of period to semi-major axis
        
        Args:
            P (astropy Quantity): list of period with unit (day)
            Ms (astropy Quantity): stellar mass
        
        Returns:
            a (astropy Quantity): list of semi-major axis with unit (AU)
        '''
        a = (P**2*const.G*Ms/(4.*np.pi**2))**(1./3.)
        return a.to('AU')
    
    def P_to_a(self):
        '''Converts period to semi-major axis for each spectral type given
        
        This method assumes there is one grid for period'''
        self.aData['input'] = False
        self.aData['unit'] = 'AU'
        self.aData['scale'] = self.PData['scale']
        self.aData['range'] = {}
        # get ranges based on Teff based mass
        tmp_dict = {}
        for i in xrange(len(self.starData['type'])):
            Ms = self.get_Ms_from_Teff(self.starData['Teff'][i])
            P = np.array(self.PData['range'])*getattr(u,self.PData['unit'])
            tmp_a = self.get_a_from_P(P,Ms)
            tmp_dict[self.starData['type'][i]] = tmp_a.to('AU').value
        self.aData['range'].update(tmp_dict)
        
    def get_P_from_a(self,a,Ms):
        '''Converts list of semi-major axis to period
        
        Args:
            a (astropy Quantity): list of semi-major axis with unit (AU)
            Ms (astropy Quantity): stellar mass
            
        Returns:
            P (astropy Quantity): list of periods with unit (day)
        '''
        P = 2.*np.pi*np.sqrt(a**3/(const.G*Ms))
        return P.to('day')
        
    def a_to_P(self):
        '''Converts semi-major axis to period for each spectral type given
        
        This method assumes there is one grid for semi-major axis'''
        self.PData['input'] = False
        self.PData['unit'] = 'day'
        self.PData['scale'] = self.aData['scale']
        self.PData['range'] = {}
        # get ranges based on Teff based mass
        tmp_dict = {}
        for i in xrange(len(self.starData['type'])):
            Ms = self.get_Ms_from_Teff(self.starData['Teff'][i])
            a = np.array(self.aData['range'])*getattr(u,self.aData['unit'])
            tmp_P = self.get_P_from_a(a,Ms)
            tmp_dict[self.starData['type'][i]] = tmp_P.to('day').value
        self.PData['range'].update(tmp_dict)
        
    def get_Ms_from_Teff(self,Trange):
        '''Finds average stellar mass for Teff range'''
        Ms = self.Msun_from_Teff.integral(Trange[0],Trange[1])/(Trange[1]-Trange[0])*u.M_sun
        return Ms
    
    def get_Rp_from_M(self,M):
        '''Converts planetary mass to radius
        
        Args:
            M (ndarray): list of masses in earthMass
        
        Returns:
            Rp (ndarray): list of radii in earthRad
        '''
        # get Forecaster coefficients
        C, S, T, R = self.Forecaster_coeffs()
        # if single value, cast to array
        M = np.array(M, ndmin=1, copy=False)
        Rp = np.zeros(M.shape)
        for j in xrange(len(T)-1):
            vals = (M > T[j])&(M <= T[j+1])
            Rp[vals] = 10.0**C[j]*M[vals]**S[j]
        return Rp
        
    def M_to_Rp(self):
        '''Converts mass to planetary radius for each spectral type given
        
        This method uses the Forecaster model to convert mass to radius'''
        self.RpData['input'] = False
        self.RpData['unit'] = 'earthRad'
        self.RpData['scale'] = 'linear'
        # get range of Rp from mass
        M = np.array(self.MsiniData['range'])*getattr(u,self.MsiniData['unit']).to('earthMass')
        self.RpData['range'] = self.get_Rp_from_M(M)
    
    def get_M_from_Rp(self,Rp):
        '''Converts planetary radius to mass
        
        Args:
            Rp (ndarray): list of planetary radii in earthRad
            
        Returns:
            M (ndarray): list of planetary masses in earthMass
        '''
        # get Forecaster coefficients
        C, S, T, R = self.Forecaster_coeffs()
        # if single value, cast to array
        Rp = np.array(Rp, ndmin=1, copy=False)
        M = np.zeros(Rp.shape)
        for j in xrange(len(T)-1):
            vals = (Rp>R[j])&(Rp<=R[j+1])
            M[vals] = 10.0**((np.log10(Rp[vals])-C[j])/S[j])
        return M
    
    def Rp_to_M(self):
        '''Converts planetary radius to mass for each spectral type given
        
        This method uses the Forecaster model to convert radius to mass'''
        self.MsiniData['input'] = False
        self.MsiniData['unit'] = 'earthMass'
        self.MsiniData['scale'] = 'log'
        # get range of mass from Rp
        Rp = np.array(self.RpData['range'])*getattr(u,self.RpData['unit']).to('earthRad')
        self.MsiniData['range'] = self.get_M_from_Rp(Rp)
        
    def Forecaster_coeffs(self):
        '''Determines coefficients from Forecaster modified to transition point
        at Saturn mass and slight incline past Jupiter mass from Bashi et al. 2017'''
        # initial values
        S = np.array([0.2790,0,0,0,0.881]) # exponent
        C = np.array([np.log10(1.008), 0, 0, 0, 0]) # coefficient
        T = np.array([0.,2.04,95.16,(u.M_jupiter).to(u.M_earth),((0.0800*u.M_sun).to(u.M_earth)).value,np.inf]) # mass break points
        Rj = u.R_jupiter.to(u.R_earth)
        Rs = 8.522 # Saturn radius
        # between T[1] and Saturn
        S[1] = (np.log10(Rs) - (C[0] + np.log10(T[1])*S[0]))/(np.log10(T[2]) - np.log10(T[1]))
        C[1] = np.log10(Rs) - np.log10(T[2])*S[1]
        # between Saturn and Jupiter
        S[2] = (np.log10(Rj) - np.log10(Rs))/(np.log10(T[3]) - np.log10(T[2]))
        C[2] = np.log10(Rj) - np.log10(T[3])*S[2]
        # between Jupiter and stellar mass
        S[3] = 0.01
        C[3] = np.log10(Rj) - np.log10(T[3])*S[3]
        # above stellar mass
        Rstell = 10.**(C[3])*T[4]**S[3]
        C[4] = np.log10(Rstell) - np.log10(T[4])*S[4]
        
        # get radius break points
        R = np.zeros(T.shape)
        R[1:] = 10.**C*T[1:]**S
        
        return C, S, T, R
    
    def plot_allPRp(self):
        '''Plots occurrence rates for each spectral type with axes of Period 
        and Planetary Radius and saves them to plots/name folder'''
        for onetype in self.starData['type']:
            self.plot_onePRp(onetype)
    
    def plot_allaRp(self):
        '''Plots occurrence rates for each spectral type with axes of 
        Semi-Major Axis and Planetary Radius and saves them to plots/name'''
        for onetype in self.starData['type']:
            self.plot_oneaRp(onetype)
    
    def plot_allPM(self):
        '''Plots occurrence rates for each spectral type with axes of Period
        and Mass and saves them to plots/name'''
        for onetype in self.starData['type']:
            self.plot_onePM(onetype)
            
    def plot_allaM(self):
        '''Plots occurrence rates for each spectral type with axes of 
        Semi-Major Axis and Mass and saves them to plots/name'''
        for onetype in self.starData['type']:
            self.plot_oneaM(onetype)

    def plot_onePRp(self, typekey):
        '''Plots occurrence rates for a spectral type with axes of Period
        and planetary radius and saves plot to plots folder
        (INCLUDE TEXT ON PLOT?)
        Args:
            typekey (string): stellar type'''
        # use TeX fonts
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(12,8))
        ax.set_title('{} Star Occurrence Rate [\%] from {}'.format(typekey, self.fname['title']),fontsize=16)
        ax.set_xlabel('Orbital Period [days]',fontsize=16)
        ax.set_ylabel(r'Planet Radius [R$_\bigoplus$]',fontsize=16)
        # edges of the plot
        if self.PData['input']:
            x = np.array(self.PData['range'])*getattr(u,self.PData['unit']).to('day')
        else:
            x = np.array(self.PData['range'][typekey])*getattr(u,self.PData['unit']).to('day')
        y = np.array(self.RpData['range'])*getattr(u,self.RpData['unit']).to('R_earth')
        xlabel = ['{0:.3g}'.format(_x) for _x in x]
        ylabel = ['{0:.3g}'.format(_y) for _y in y]
        # plot values
        eta = self.Etas[typekey]['eta']
        eta = np.ma.masked_where(np.isnan(eta), eta)
        cmap = plt.cm.Blues_r
        cmap.set_under('k')
        cmap.set_over('w')
        cmap.set_bad('k',1)
        a = ax.pcolormesh(x,y,100*eta,norm=colors.LogNorm(vmin=1e-3,vmax=1e2),rasterized=True,edgecolor='none',cmap=cmap)
        c = fig.colorbar(a,ticks=[0.001,0.01,0.1,1,10,100])
        c.ax.set_yticklabels(['{} \%'.format(10.**(vv)) for vv in range(-3,3)],fontsize=14)
        # scale for axes
        if self.PData['scale'] == 'log':
            ax.set_xscale('log')
        if self.RpData['scale'] == 'log':
            ax.set_yscale('log')

        ax.set_xticks(x)
        ax.set_xticklabels(xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabel)
        ax.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=14)
        ax.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
        fig.show()
        # save figure
        folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'plots')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        plotfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'plots',self.fname['name'])
        if not os.path.isdir(plotfolder):
            os.mkdir(plotfolder)
        plotpath = os.path.join(plotfolder,typekey+'_eta_PRp.png')
        fig.savefig(plotpath, format='png', bbox_inches='tight',pad_inches=0.1)
    
    def plot_oneaRp(self, typekey):
        '''Plots occurrence rates for a spectral type with axes of Period
        and planetary radius and saves plot to plots folder
        (INCLUDE TEXT ON PLOT?)
        Args:
            typekey (string): stellar type'''
        # use TeX fonts
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(12,8))
        ax.set_title('{} Star Occurrence Rate [\%] from {}'.format(typekey, self.fname['title']),fontsize=16)
        ax.set_xlabel('Semi-Major Axis [AU]',fontsize=16)
        ax.set_ylabel(r'Planet Radius [R$_\bigoplus$]',fontsize=16)
        # edges of the plot
        if self.aData['input']:
            x = np.array(self.aData['range'])*getattr(u,self.aData['unit']).to('AU')
        else:
            x = np.array(self.aData['range'][typekey])*getattr(u,self.aData['unit']).to('AU')
        y = np.array(self.RpData['range'])*getattr(u,self.RpData['unit']).to('R_earth')
        xlabel = ['{0:.3g}'.format(_x) for _x in x]
        ylabel = ['{0:.2g}'.format(_y) for _y in y]
        # plot values
        eta = self.Etas[typekey]['eta']
        eta = np.ma.masked_where(np.isnan(eta), eta)
        cmap = plt.cm.Blues_r
        cmap.set_under('k')
        cmap.set_over('w')
        cmap.set_bad('k',1)
        a = ax.pcolormesh(x,y,100*eta,norm=colors.LogNorm(vmin=1e-3,vmax=1e2),rasterized=True,edgecolor='none',cmap=cmap)
        c = fig.colorbar(a,ticks=[0.001,0.01,0.1,1,10,100])
        c.ax.set_yticklabels(['{} \%'.format(10.**(vv)) for vv in range(-3,3)],fontsize=14)
        # scale for axes
        if self.aData['scale'] == 'log':
            ax.set_xscale('log')
        if self.RpData['scale'] == 'log':
            ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabel)
        ax.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=14)
        ax.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
        fig.show()
        # save figure
        folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'plots')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        plotfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'plots',self.fname['name'])
        if not os.path.isdir(plotfolder):
            os.mkdir(plotfolder)
        plotpath = os.path.join(plotfolder,typekey+'_eta_aRp.png')
        fig.savefig(plotpath, format='png', bbox_inches='tight',pad_inches=0.1)
        
    def plot_oneaM(self, typekey):
        '''Plots occurrence rates for a spectral type with axes of semi-major
        axis and mass and saves plot to plots folder
        (INCLUDE TEXT ON PLOT?)
        Args:
            typekey (string): stellar type'''
        # use TeX fonts
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(12,8))
        ax.set_title('{} Star Occurrence Rate [\%] from {}'.format(typekey, self.fname['title']),fontsize=16)
        ax.set_xlabel('Semi-Major Axis [AU]',fontsize=16)
        ax.set_ylabel(r'Planet Mass [M$_\bigoplus$]',fontsize=16)
        # edges of the plot
        if self.aData['input']:
            x = np.array(self.aData['range'])*getattr(u,self.aData['unit']).to('AU')
        else:
            x = np.array(self.aData['range'][typekey])*getattr(u,self.aData['unit']).to('AU')
        y = np.array(self.MsiniData['range'])*getattr(u,self.MsiniData['unit']).to('earthMass')
        xlabel = ['{:.3g}'.format(_x) for _x in x]
        ylabel = ['{:.2g}'.format(_y) for _y in y]
        # plot values
        eta = self.Etas[typekey]['eta']
        eta = np.ma.masked_where(np.isnan(eta), eta)
        cmap = plt.cm.Blues_r
        cmap.set_under('k')
        cmap.set_over('w')
        cmap.set_bad('k',1)
        a = ax.pcolormesh(x,y,100*eta,norm=colors.LogNorm(vmin=1e-3,vmax=1e2),rasterized=True,edgecolor='none',cmap=cmap)
        c = fig.colorbar(a,ticks=[0.001,0.01,0.1,1,10,100])
        c.ax.set_yticklabels(['{} \%'.format(10.**(vv)) for vv in range(-3,3)],fontsize=14)
        # scale for axes
        if self.aData['scale'] == 'log':
            ax.set_xscale('log')
        if self.MsiniData['scale'] == 'log':
            ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabel)
        ax.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=14)
        ax.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
        fig.show()
        # save figure
        folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'plots')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        plotfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'plots',self.fname['name'])
        if not os.path.isdir(plotfolder):
            os.mkdir(plotfolder)
        plotpath = os.path.join(plotfolder,typekey+'_eta_aM.png')
        fig.savefig(plotpath, format='png', bbox_inches='tight',pad_inches=0.1)
        
    def plot_onePM(self, typekey):
        '''Plots occurrence rates for a spectral type with axes of period and
        mass and saves plot to plots folder
        (INCLUDE TEXT ON PLOT?)
        Args:
            typekey (string): stellar type'''
        # use TeX fonts
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        fig, ax = plt.subplots(figsize=(12,8))
        ax.set_title('{} Star Occurrence Rate [\%] from {}'.format(typekey, self.fname['title']),fontsize=16)
        ax.set_xlabel('Orbital Period [days]',fontsize=16)
        ax.set_ylabel(r'Planet Mass [M$_\bigoplus$]',fontsize=16)
        # edges of the plot
        if self.PData['input']:
            x = np.array(self.PData['range'])*getattr(u,self.PData['unit']).to('day')
        else:
            x = np.array(self.PData['range'][typekey])*getattr(u,self.PData['unit']).to('day')
        y = np.array(self.MsiniData['range'])*getattr(u,self.MsiniData['unit']).to('earthMass')
        xlabel = ['{:.3g}'.format(_x) for _x in x]
        ylabel = ['{:.2g}'.format(_y) for _y in y]
        # plot values
        eta = self.Etas[typekey]['eta']
        eta = np.ma.masked_where(np.isnan(eta), eta)
        cmap = plt.cm.Blues_r
        cmap.set_under('k')
        cmap.set_over('w')
        cmap.set_bad('k',1)
        a = ax.pcolormesh(x,y,100*eta,norm=colors.LogNorm(vmin=1e-3,vmax=1e2),rasterized=True,edgecolor='none',cmap=cmap)
        c = fig.colorbar(a,ticks=[0.001,0.01,0.1,1,10,100])
        c.ax.set_yticklabels(['{} \%'.format(10.**(vv)) for vv in range(-3,3)],fontsize=14)
        # scale for axes
        if self.aData['scale'] == 'log':
            ax.set_xscale('log')
        if self.MsiniData['scale'] == 'log':
            ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(ylabel)
        ax.tick_params(axis='both', bottom='on', top='off', right='off', left='on', which='major', labelsize=14)
        ax.tick_params(axis='both', bottom='off', top='off', right='off', left='off', which='minor')
        fig.show()
        # save figure
        folder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'plots')
        if not os.path.isdir(folder):
            os.mkdir(folder)
        plotfolder = os.path.join(os.path.abspath(os.path.dirname(__file__)),'plots',self.fname['name'])
        if not os.path.isdir(plotfolder):
            os.mkdir(plotfolder)
        plotpath = os.path.join(plotfolder,typekey+'_eta_PM.png')
        fig.savefig(plotpath, format='png', bbox_inches='tight',pad_inches=0.1)