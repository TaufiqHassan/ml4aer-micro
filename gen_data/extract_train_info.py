import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import argparse

from sklearn.model_selection import train_test_split

class gen_TrainData_DistPlot(object):
    def __init__(self,datadir,data_files,**kwargs):
        self.datadir = datadir
        self.data_files = data_files
        self.timeslice = kwargs.get('timeslice', [0,1])
        self.levslice = kwargs.get('levslice', [0,72])
        self.varlist = kwargs.get('varlist', [])
        self.atmos_vars = kwargs.get('atmos_vars', [])
        self.cloudfree = kwargs.get('cloudfree', True)
        self.outdir = kwargs.get('outdir', '/pscratch/sd/h/hass877/MLdir/E3SM_data/')
        self.out = kwargs.get('out', True)
        self.test_size = kwargs.get('test_size', 0.66)
        self.logscale = kwargs.get('logscale', False)
        self.endstring = kwargs.get('endstring', '')
        
    def _get_data(self):
        data_list = list(map(lambda x: self.datadir + x, self.data_files))
        data = xr.open_mfdataset(data_list)

        ## Extract time/level slices as necessary
        if self.timeslice:
            data = data.isel(time=slice(self.timeslice[0],self.timeslice[1]))
        if self.levslice:
            data = data.isel(lev=slice(self.levslice[0],self.levslice[1]))

        ## Based on reduced variable list from https://confluence.pnnl.gov/confluence/display/PDEMLC/Reduced+input+list
        # Atmospheric condition variables
        if self.atmos_vars == []:
            self.atmos_vars = ['i_qh2o','i_pdel','i_pmid','i_T','i_zm','i_pblh',\
                          'i_a1_dgnum','i_a2_dgnum','i_a3_dgnum','i_a4_dgnum',\
                          'i_a1_dgnumwet','i_a2_dgnumwet','i_a3_dgnumwet','i_a4_dgnumwet',\
                          'i_a1_wetdens','i_a2_wetdens','i_a3_wetdens','i_a4_wetdens',\
                          'd_vmr_H2SO4_GsChem']
        # Mixing ratios
        if self.varlist == []:
            self.varlist = ['bc_a1','bc_a4',\
                       'pom_a1','pom_a4',\
                       'so4_a1','so4_a2','so4_a3',\
                       'mom_a1','mom_a2','mom_a4',\
                       'ncl_a1','ncl_a2',\
                       'soa_a1','soa_a2','soa_a3',\
                       'num_a1','num_a2','num_a4',\
                       'H2SO4','SOAG']

        ## Feature creation from zm & pblh
        zm = data['i_zm']
        pbl = data['i_pblh']
        max_pbl = xr.where(pbl > 100.0, pbl, 100.0)
        npbl = xr.full_like(zm, 0.0)
        npbl = xr.where(zm <= max_pbl, 1.0, npbl)
        data['i_pblh'] = npbl
        ## Feature creation for d_vmr_H2SO4_preGsChem
        data['d_vmr_H2SO4_GsChem'] = data['i_vmr_H2SO4_preAerMic'] - data['i_vmr_H2SO4_preGsChem']

        ## Input/Feature variables
        e3smInVars = ['i_vmr_'+v+'_preAerMic' for v in self.varlist]
        e3sm_in_vars = self.atmos_vars + e3smInVars
        ## Output Target variables
        e3sm_out_vars = ['o_vmr_'+v+'_pstAerMic' for v in self.varlist]

        # Load actual data for these variables
        data_out = data[e3sm_out_vars]
        data_in = data[e3sm_in_vars]

        ## Maskout cloudy grids
        if self.cloudfree:
            cld = data['i_cldfr'].load()
            data_in = data_in.where(cld==0)
            data_out = data_out.where(cld==0)
        
        return data_in, data_out
    
    def _flatten_data(self):
        data_in, data_out = self._get_data()
        flattened_data_in = []
        flattened_data_out = []
        for vout in data_out.data_vars:
            flattened_data_out.append(data_out[vout].values.flatten())
        for vin in data_in.data_vars:
            flattened_data_in.append(data_in[vin].values.flatten())
        flattened_data_in = np.stack(flattened_data_in, axis=1)
        flattened_data_out = np.stack(flattened_data_out, axis=1)
        print(flattened_data_in.shape, flattened_data_out.shape)
        return flattened_data_in, flattened_data_out
    
    def _gen_train_data(self):
        flattened_data_in, flattened_data_out = self._flatten_data()
        # Remove rows with NaN values in flattened_data_in
        mask = ~np.isnan(flattened_data_in).any(axis=1)
        flattened_data_in = flattened_data_in[mask]
        flattened_data_out = flattened_data_out[mask]
        print(flattened_data_in.shape,flattened_data_out.shape)
        # Split the data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = train_test_split(flattened_data_in, flattened_data_out, test_size=self.test_size, random_state=42)
        # Split the remaining into validation and test sets
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        # Check the sizes of the datasets
        print(f"Training data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
        print(f"Validation data: X_val shape = {X_val.shape}, y_val shape = {y_val.shape}")
        print(f"Testing data: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")
        if self.out:
            np.save(self.outdir+'X_train.npy',X_train)
            np.save(self.outdir+'y_train.npy',y_train)
            np.save(self.outdir+'X_val.npy',X_val)
            np.save(self.outdir+'y_val.npy',y_val)
            np.save(self.outdir+'X_test.npy',X_test)
            np.save(self.outdir+'y_test.npy',y_test)
        else:
            return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_stats(self,val):
        df = pd.DataFrame()
        mean = []
        maxvals = []
        minvals = []
        stdvals = []
        for i in range(len(self.varlist)):
            mean.append(val[:,i].mean())
            maxvals.append(val[:,i].max())
            minvals.append(val[:,i].min())
            stdvals.append(val[:,i].std())
        df['Target'] = self.varlist
        df['Mean'] = mean
        df['Max'] = maxvals
        df['Min'] = minvals
        df['Std'] = stdvals
        return df
    
    def _get_2d_hist_single(self,data,index,ax):
        for i in range(72):
            intData = data[21600*(i):21600*(i+1),index]
            mask = ~np.isnan(intData)
            intData = intData[mask]
            if all(intData==0):
                ax.scatter(0,i,marker='_',zorder=1,c='#0d0887')
            elif self.logscale:
                self.endstring = 'logscale'
                zero_values = np.where(intData == 0)[0]
                
                positive_values = np.where(intData > 0)[0]
                hh_pos = np.histogram(intData[positive_values],bins=500)
                aa_pos = (hh_pos[1][1:]-hh_pos[1][:-1])/2
                pl = ax.scatter(np.log10(hh_pos[1][:-1]+aa_pos[0]),np.zeros(len(hh_pos[0]))+i,c=hh_pos[0],cmap='plasma',norm = colors.LogNorm(vmin=0.05,vmax=1e6),s=5,marker=',',zorder=5)
                
                negative_values = np.where(intData < 0)[0]
                hh_neg = np.histogram(intData[negative_values],bins=500)
                aa_neg = (hh_neg[1][1:]-hh_neg[1][:-1])/2
                pl = ax.scatter(-1*np.log10(np.abs(hh_neg[1][:-1]+aa_neg[0])),np.zeros(len(hh_neg[0]))+i,c=hh_neg[0],cmap='plasma',norm = colors.LogNorm(vmin=0.05,vmax=1e6),s=5,marker=',',zorder=5)
            else:
                hh = np.histogram(intData,bins=500)
                aa=(hh[1][1:]-hh[1][:-1])/2
                pl = ax.scatter(hh[1][:-1]+aa[0],np.zeros(len(hh[0]))+i,c=hh[0],cmap='plasma',norm = colors.LogNorm(vmin=0.05,vmax=1e6),s=5,marker=',',zorder=5)
                ax.tick_params(labelsize=15)
            plt.ylim([0,72])
            ax.grid(zorder=2,color='lightgray')
        ax.invert_yaxis()
        return pl
    
    def _get_2d_hist(self,data,name):
        mask = ~np.isnan(data).any(axis=1)
        intData = data[mask]
        df = self.get_stats(intData)
        
        fig = plt.figure(figsize=(24,18))
        for i in range(data.shape[1]):
            ax = plt.subplot(5,4,i+1)
            pl = self._get_2d_hist_single(data,i,ax)
            ax.text(0.005, 1.03, self.varlist[i], size=20, transform=ax.transAxes)
            ax.text(0.4, 1.08, 'Mean: '+str('{:.2e}'.format(df.iloc[i].values[1])), size=12, transform=ax.transAxes)
            ax.text(0.4, 1.01, 'STD: '+str('{:.2e}'.format(df.iloc[i].values[4])), size=12, transform=ax.transAxes)
            ax.text(0.75, 1.08, 'Max: '+str('{:.2e}'.format(df.iloc[i].values[2]).replace('e+00','')), size=12, transform=ax.transAxes)
            ax.text(0.75, 1.01, 'Min: '+str('{:.2e}'.format(df.iloc[i].values[3]).replace('e+00','')), size=12, transform=ax.transAxes)
            
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        cax = fig.add_axes([0.01,0.01,0.99,0.03])
        ranges = [1,10,1e2,1e3,1e4,1e5,1e6,1e7]
        cbar = fig.colorbar(pl,cax=cax,orientation='horizontal',extend='neither',ticks=ranges,fraction=0.08)
        ranges = list(map(lambda x: '{:.0e}'.format(x).replace('e+00','.0'),ranges))
        cbar.ax.set_xticklabels(labels=ranges,size=20 )
        cbar.set_label(label='# of samples',size=20)
        plt.savefig(name+'_'+self.endstring+'.png',format='png',dpi=300,bbox_inches='tight',pad_inches=0.1)
        
    def get_all_hist(self):
        flattened_data_in, flattened_data_out = self._flatten_data()
        diff = (flattened_data_out-flattened_data_in[:,len(self.atmos_vars):])
        self._get_2d_hist(flattened_data_in[:,len(self.atmos_vars):],'Input_features')
        self._get_2d_hist(diff[:,:],'Output_targets')
    


def main():
    parser = argparse.ArgumentParser(description="Generate training data and plot distribution histograms for E3SM data.")
    
    # Add arguments for the CLI
    parser.add_argument('--datadir', type=str, required=True, help="Directory where the data files are located.")
    parser.add_argument('--data_files', type=str, nargs='+', required=True, help="List of data file names.")
    parser.add_argument('--timeslice', type=int, nargs=2, default=[0, 1], help="Time slice as a list of two integers (start, end).")
    parser.add_argument('--levslice', type=int, nargs=2, default=[0, 72], help="Level slice as a list of two integers (start, end).")
    parser.add_argument('--varlist', type=str, nargs='*', default=[], help="List of mixing ratio variables.")
    parser.add_argument('--atmos_vars', type=str, nargs='*', default=[], help="List of atmospheric condition variables.")
    parser.add_argument('--cloudfree', type=bool, default=True, help="Whether to mask out cloudy grids (default: True).")
    parser.add_argument('--outdir', type=str, default='/pscratch/sd/h/hass877/MLdir/E3SM_data/', help="Output directory to save numpy arrays and plots.")
    parser.add_argument('--out', type=bool, default=True, help="Whether to save the output files (default: True).")
    parser.add_argument('--test_size', type=float, default=0.66, help="Test set size as a proportion (default: 0.66).")
    parser.add_argument('--logscale', type=bool, default=False, help="Whether to use log scale in the plots (default: False).")
    parser.add_argument('--endstring', type=str, default='', help="Suffix for the output file names.")

    # Choose which method to run
    parser.add_argument('--action', type=str, choices=['gen_data', 'gen_hist', 'gen_all_hist'], required=True, help="Choose an action: 'gen_data' to generate training data or 'get_hist' to generate 2D histograms or 'gen_all_hist' to generate two 2D histograms for state variables and tendencies.")
    
    args = parser.parse_args()
    
    # Initialize the class with parsed arguments
    generator = gen_TrainData_DistPlot(
        datadir=args.datadir,
        data_files=args.data_files,
        timeslice=args.timeslice,
        levslice=args.levslice,
        varlist=args.varlist,
        atmos_vars=args.atmos_vars,
        cloudfree=args.cloudfree,
        outdir=args.outdir,
        out=args.out,
        test_size=args.test_size,
        logscale=args.logscale,
        endstring=args.endstring
    )
    
    # Perform the selected action
    if args.action == 'gen_data':
        generator._gen_train_data()
    elif args.action == 'gen_hist':
        generator._get_2d_hist()
    elif args.action == 'gen_all_hist':
        generator.get_all_hist()
        
if __name__ == "__main__":
    main()

