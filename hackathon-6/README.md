## Examples of NWB file creation and querying for [Hackathon 6](https://github.com/NeurodataWithoutBorders/nwb_hackathons/tree/master/HCK06_2019_Janelia) at Janelia, May 2019

Running against pynwb 'dev' branch as of May 10, 2019

In addition to the pynwb requirements, you will need:
 `networkx matplotlib scipy python-dateutil`

* Frank Lab dataset from CRCNS:
  * [Download link](https://portal.nersc.gov/project/crcns/download/hc-6) (requires CRCNS.org account) 
  * [Dataset about page](https://crcns.org/data-sets/hc/hc-6/about-hc-5)
  * [Bon04.nwb](https://www.dropbox.com/s/92jkkse2c7lm7qe/bon04.nwb?dl=0) (~4GB), the file created by the hackathon-6/create_franklab_nwbfile.ipynb, which can be used to run the queries in the place_field_with_queries.ipynb notebook.
* [Allen Institute pre-release NWB files](http://download.alleninstitute.org/informatics-archive/prerelease/)
  * [ecephys_session_785402239.nwb](http://download.alleninstitute.org/informatics-archive/prerelease/ecephys_session_785402239.nwb.bz2) (~0.6GB). File to use with frank_lab_queries_allen_data.ipynb.
