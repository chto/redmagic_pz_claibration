import fitsio
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import fitting_scheme as fs
from multiprocessing import Pool
import os
import sys
try:
  from schwimmbad import MPIPool
except:
  print("cannot load MPIpool")
import util_chto
import argparse
def worker(ind):
  print("working on : {0}".format(ind))
  sys.stdout.flush()
  if os.path.isfile(outdir+"redmaic_ind_{0}_lin_emgm_param.npy".format(ind)):
    print("skipping .... {0}".format(ind))
    return
  z = zlist[ind]
  if z<0.65:
      cat = cat_all['highdense']
  elif z<0.8:
      cat = cat_all['highlum']
  else:
      cat = cat_all['higherlum'] 
  mask = np.abs(cat['ZREDMAGIC']-z)<0.01
  bins=np.linspace(0.1,1.1, 200)
  bins_center = 0.5*(bins[1:]+bins[:-1])
  histogram, _ = np.histogram(cat['ZSPEC'][mask], bins=bins, normed=True)
  histogram_test, _ = np.histogram(cat['ZSPEC'][mask], bins=bins, normed=False)
  hist_err = histogram/np.sqrt(histogram_test)    
  x = cat['ZSPEC'][mask].reshape(-1,1)
  #Fitting 
  emgm = fs.EM_GMM(ngaussian=2)
  emgm.fit(x)
  param = emgm.param
  #Plot
  plt.figure()
  plt.title("$z_{{redmagic}}$={0}".format(z))
  plt.errorbar(bins_center, histogram, yerr=hist_err, fmt="o", markersize='1')
  plt.plot(bins_center, fs.fit_function(bins_center.reshape(-1,1), param), lw=1.5)
  plt.xlabel("ZSPEC")
  plt.ylabel("normed hist")
  plt.axvline(z, c="r")
  plt.savefig(outdir+"redmagic_ind_{0}_lin_emgm.png".format(ind))
  np.save(outdir+"redmaic_ind_{0}_lin_emgm_param.npy".format(ind), param)
  plt.clf()
  print("working on : {0} done".format(ind))
if __name__ == "__main__":
  ###Read argument
  parser = argparse.ArgumentParser(description='run redmagic_pz_calibration')
  parser.add_argument("--paramfile", help="YAML configuration file")
  args = parser.parse_args()
  params = util_chto.chto_yamlload(args.paramfile)
  ###Read catalog
  cat_all = {}
  cat_all['highdense'] = fitsio.read(params['redmagic_highdense'])
  cat_all['highlum'] = fitsio.read(params['redmagic_highlum'])
  cat_all['higherlum'] = fitsio.read(params['redmagic_higherlum'])
  outdir = params['outputdir'] 
  zlist = np.linspace(*params['zbins'])
  #p = Pool(5)
  mpipool = MPIPool()
  if not mpipool.is_master():
    mpipool.wait()
    sys.exit(0)
  mpipool.map(worker, np.arange(len(zlist)))
  mpipool.close()

  
 
