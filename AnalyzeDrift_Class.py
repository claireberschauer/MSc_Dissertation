""" AnalyzeDrift Module

This module contains a function to set up a Pandas data frame of Calibration
data in the UDD ROOT format. It also has the AnalyzeDrift class, which defines 
a function to set up the drift_df data frame with drift time, drift radius, and 
cell type data. The class also contains a dictionary that contains all of the 
parameters needed to run Betsy's drift model and lists to define cell types and 
tag dead cells based on location with the tracker.

Developed by Claire Berschauer for the SuperNEMO collaboration.
Last edited: 05 September 2023
"""

# Imports
import numpy as np
import time as time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import uproot
import cvxpy as cp
import random

def process_data(file_name, num_events=1000, all_events=False, verbose=False, 
				 print_time=True, branch='SimData;1', name='events.csv', 
				 index=False):

    """ This function takes UDD ROOT format data (.root) and converts it to a 
    pandas data frame style .csv file.

    :param file_name: Raw (commissioning) data, must be a UDD ROOT file.
    :type file_name: str

    :param num_events: Integer that indicates how many events you would like to 
    process. The default 1000 events takes about 30 minutes to run. Default is 
    1000 events.
    :type num_events: int, optional

    :param all_events: Option to process all events. WARNING: this function 
    takes a long time to run. If you choose all_events=True, the num_events 
    parameter will become obsolete. Default is False.
    :type all_events: bool, optional

    :param verbose: Indicates whether you would like the function to print "n 
    of N complete" after each event has been processed. This will also print 
    the time it takes for the function to run. Default is False.
    :type verbose: bool, optional

    :param print_time: To print the run time without printing a message after 
    each event iteration like the verbose kwarg does. Default is True. 
    :type print_time: bool, optional

    :param branch: Defines the branch within the ROOT file to access. Default 
    is 'SimData;1'.
    :type branch: str, optional

    :param name: Determines what the resulting csv file will be called, default 
    is 'events.csv'. Default is 'events.csv'.
    :type name: str, optional

    :param index: Default of False does not include the index as a column in 
    the csv file. Default is False.
    :type index: bool, optional
    """
    
    start = time.time()

    data = uproot.open(file_name)
    id_branch = data[branch]['digitracker.id'].array(library='np')

    if all_events is True:
        n_events = len(id_branch)
    else:
        n_events = num_events

    # initializing lists to be added as columns in the final data frame
    event_nums = []
    ids = []
    sides = []
    layers = []
    columns = []
    R0s = []
    calo_times = []
    rc_times = []

    # loop to add all the datapoints into entries in the lists -- converts the 
    # root 'branch' format to a single list for each variable (so each hit's 
    # data is stored in a single row in the data frame)
    for n in np.arange(n_events):
        # indicates how many hits occur in the given event
        id_list = np.array(id_branch[n], dtype='int')
        num_hits = len(id_list)
        
        # calling information from the data file -- one array per variable per 
        # event and appending data to respective lists
        try: 
            ev_R0s = data['SimData;1']['digitracker.anodetimestampR0'].array(
            		 library='np')[n]  # R0 timestamps from anode 
            # [clock ticks], tail end of drift time
            R0s.extend([x for [x] in ev_R0s])  # this format since ev_R0s is of 
            # the form [[#], [#], ...], so need to get rid of extra brackets
            
        except:
            R0s.extend([np.nan] * num_hits)  # if an error occurs while 
            # extracting R0 data, which sometimes happens, this adds num_hits 
            # nan values to the R0s list to prevent mismatched list lengths
        
        try:
            calo_times.extend([data['SimData;1']['digicalo.timestamp'].array( 
            library='np')[n][0]] * num_hits)  # using the first calorimeter 
            # timestamp arbitrarily as the trigger time. Note: the calorimeter 
            # clock runs twice as fast as the tracker clock.
            rc_times.extend([data['SimData;1']['digicalo.rising_cell'].array( 
            library='np')[n][0]] * num_hits)  # needed for pulse start time

        except:
            calo_times.extend([np.nan] * num_hits)  # once again adding nan 
            # values if an exception occurs
            rc_times.extend([np.nan] * num_hits)
        
        
        ev_sides = data['SimData;1']['digitracker.side'].array( 
        		   library='np')[n]  # French side = 1, Italian = 0 (Since 
                   # France is #1!)
        ev_layers = data['SimData;1']['digitracker.layer'].array(
        				 library='np')[n]  # Layer 0 near source foil, layer 8 
                         # by calo wall
        ev_columns = data['SimData;1']['digitracker.column'].array( 
        			 library='np')[n]  # Column 0 on mountain side, col 112 on 
                     # tunnel side
        ev_ids = data['SimData;1']['digitracker.id'].array(library='np')[n] 
        # line above is for Event IDs

        # use .extend() to append lists rather than individual items
        event_nums.extend([n] * num_hits)  # all hits in each iteration should 
        # have the same event ID, so add num_hits IDs to the list
        sides.extend(ev_sides)
        layers.extend(ev_layers)
        columns.extend(ev_columns)
        ids.extend(ev_ids)
        
        # Prints a message after every iteration; nice for low numbers of events
        if verbose is True:
            print(n + 1, 'of', n_events, 'complete')
    
    # Creates a dictionary of the data lists with corresponding names to be 
    # used as column names in the data frame
    data={'Event':event_nums, 'ID':ids, 'Side':sides, 'Column':columns, 
    	  'Layer':layers, 'R0':R0s, 'Calo_time':calo_times, 
    	  'Rising_cell_time':rc_times
    	  }
    
    # Makes a data frame of the data and saves it as a .csv file in the same 
    # folder as the program running this function
    event_df = pd.DataFrame(data)
    event_df.to_csv(name, index=index)
    
    end = time.time()
    
    # Prints runtime
    if verbose is True:
        print('Runtime:', (end - start)/60, 'minutes')
    elif print_time is True:
        print('Runtime:', (end - start)/60, 'minutes')


class AnalyzeDrift():
    """ The AnalyzeDrift class aims to streamline the process of data 
	preparation for drift radius calculation. It automatically finds drift 
	times based on timestamp data, identifies cell type based on location, and 
	calculates drift radius, compiling all relevant information in the drift_df 
	attribute. 
    
    :param file_name: Name of the input data frame. Ideally this data frame 
    will be in the form produced by the process_data function, since column 
    names are assumed to be 'Event', 'ID', 'Side', 'Column', 'Layer', 'R0', 
    'Calo_time', and 'Rising_cell_time'. Data needs to be a .csv file input. 
    :type file_name: str

    :param index: Allows the user to input a new index column if desired. 
    Default 42 indicates that the pandas.read_csv() function shouldn't specify 
    and index column.
    :type index: int, optional

    :param pressure: Tells the functions what pressure to use when selecting 
    drift model parameters. The default, 880, is closest to the actual 
    demonstrator module tracking chamber gas pressure. Parameters have been 
    calculated for pressure values of 850, 880, and 910. 
    :type pressure: int, optional
    """
    
    def __init__(self, file_name, index=42, pressure=880):
        
        # Dictionaries!
        
        # ab_vals contains information on the parameters a and b that 
        # characterize the drift moodel for radius calculation and the value tx 
        # that gives the threshold for a particle being in the 'inner' or 
        # 'outer' region of the cell. a, b, and tx are pressure-dependent, 
        # which is why each entry has three tuples of three values.
        
        # 'region_name' : [(a_850, b_850, tx_850), (a_880, b_880, tx_880), ... 
        # ... (a_910, b_910, tx_910)]
        
        self.ab_vals = {'center_in': [(8.28, -0.9, 2.95), (8.53, -0.9, 2.97), 
        							  (8.77, -0.9, 3.06)],
                        'center_out': [(3.86, -1.99, 2.95), (4.19, -1.93, 2.97),
                        			   (4.55, -1.9, 3.06)],
                        'edge_in': [(8.05, -0.9, 3.73), (8.35, -0.92, 4.15), 
                        			(8.56, -0.9, 4.12)],
                        'edge_out': [(3.34, -2.04, 3.73), (3.39, -2.07, 4.15), 
                        			 (4.03, -1.91, 4.12)],
                        'corner_in': [(7.66, -0.87, 3.34), (7.92, -0.87, 3.45), 
                        			  (8.16, -0.87, 3.59)],
                        'corner_out': [(5.18, -1.4, 3.34), (4.94, -1.48, 3.45), 
                        			   (5.25, -1.45, 3.59)]
                       }
        
        # Cell Identification!
        # entry = (side, column, layer)
        
        self.dead_cells = [(0, 1, 1), (0, 2, 1), (0, 3, 0), (0, 4, 3), 
        				   (0, 9, 0), (0, 11, 0), (0, 21, 0), (0, 56, 3), 
                           (0, 56, 4), (0, 56, 5), (0, 57, 0), (0, 63, 8), 
                           (0, 70, 0), (0, 74, 4), (0, 77, 8), (0, 84, 0), 
                           (0, 86, 8), (0, 87, 0), (0, 89, 1), (0, 101, 3), 
                           (1, 32, 0), (1, 47, 8), (1, 56, 4), (1, 56, 5),
                           (1, 73, 2), (1, 79, 6), (1, 80, 2), (1, 84, 5), 
                           (1, 91, 2), (1, 99, 8), (1, 100, 4), (1, 107, 8),
                           (1, 110, 0)]
        
        self.corner_cells = [(0, 0, 0), (0, 112, 0), (0, 0, 8), (0, 112, 8), 
        					 (1, 0, 0), (1, 112, 0), (1, 0, 8), (1, 112, 8)]

        adj_cells = []  # finding cells adjacent to dead cells

        for tpl in self.dead_cells:
            adj_1 = (tpl[0], tpl[1] + 1, tpl[2])  # one column up
            adj_2 = (tpl[0], tpl[1] - 1, tpl[2])  # one column down
            adj_3 = (tpl[0], tpl[1], tpl[2] + 1)  # one layer up
            adj_4 = (tpl[0], tpl[1], tpl[2] - 1)  # one layer down
            
            adj_cells.append(adj_1)
            adj_cells.append(adj_2)
            adj_cells.append(adj_3)
            adj_cells.append(adj_4)
            

        adj_cells_in_det = [a for a in adj_cells if a[1] in range(113) and 
                            a[2] in range(9)]  # ensuring within tracker area
            
        self.adjacent_cells = list(set(adj_cells_in_det))
        
        ec = [(0, 0, l1) for l1 in np.arange(9)] 
        ec.extend([(0, c1, 0) for c1 in np.arange(113)])
        ec.extend([(0, 112, l2) for l2 in np.arange(9)]) 
        ec.extend([(0, c2, 8) for c2 in np.arange(113)])
        ec.extend([(1, 0, l3) for l3 in np.arange(9)]) 
        ec.extend([(1, c3, 0) for c3 in np.arange(113)])
        ec.extend([(1, 112, l4) for l4 in np.arange(9)]) 
        ec.extend([(1, c4, 8) for c4 in np.arange(113)])
        ec.extend(self.adjacent_cells)  # to consider dead-cell adjacent cells 
        # as edge cells in drift radius calculations
        
        self.edge_cells = [c for c in ec if c not in self.corner_cells]  # this 
        # line removes overlaps with corner cells
        
        cc = [(s, c, l) for s in np.arange(1) for c in np.arange(113) 
        	  for l in np.arange(9)]
        cc = [c for c in cc if c not in self.corner_cells]  # removes overlaps 
        # with corner cells
        
        self.center_cells = [c for c in cc if c not in self.edge_cells]  # this 
        # line removes overlaps with edge cells
        
        # Reading the data file
        if index == 42:
            df = pd.read_csv(file_name)
        else:
            df = pd.read_csv(file_name, index=index)
        
        # Initializing other attributes of the class
        self.original_df = df
        self.pressure = pressure
        self.drift_times = None
        self.drift_radii = None
        self.params = None
        
        self.calc_drift_time()
        self.drift_df = pd.concat([self.original_df, self.drift_times], axis=1)
        self.find_region()
        self.drift_df = pd.concat([self.drift_df, self.cell_types], axis=1)
        self.calc_radius()
        self.drift_df = pd.concat([self.drift_df, self.drift_radii], axis = 1)
    
    ###########################################################################
    # Data Preparation Functions                                              #
    ###########################################################################
    
    def calc_drift_time(self):
        """ Calculates the drift time using timestamp data from the input data 
        file.
        """
        
        post_trigger = 200  # setting configured to 200 ns; this is a fixed 
        # parameter to tune the position of the pusle in the record window
        sampling_period = 0.39062000 # [ns]
        pulse_start_time = (self.original_df['Rising_cell_time'] * 
        				   sampling_period / 256)  # rising cell time is the 
        # start time measured by the FEB firmwave with constant fraction 
        # discriminatior method; it is the time when the pulse amplitude 
        # reached 25% of the maximum amplitude
        converted_calo = self.original_df['Calo_time']*6.25  # calorimeter 
        # timestamps in ns
        converted_anode = self.original_df['R0']*12.5  # first anode timestamp 
        # in ns
        
        drift_times = (converted_anode - (converted_calo - 400 +  
        			   post_trigger + pulse_start_time)) * 10**(-3)  # calculat-
        # -ion and conversion to us
        
        self.drift_times = pd.Series(data=drift_times, name='Drift_time') 
        
    def define_io(self, t_drift, region):
        """ Defines whether the particle passes through the inner or outer 
        section of the drift cell.

        :param t_drift: The measured drift time, should be a single value.
        :type t_drift: float

        :param region: The cell type. This entry can be 'center', 'edge', or 
        'corner'.
        :type region: str
        """
        
        if t_drift > self.ab_vals[region+'_in'][1][2]:
            inner = False  # indicates outer
        else:
            inner = True  # indicates inner
        
        return inner
    
    def find_region(self):
        """ Identifies whether the drift cell is edge, corner, or center based 
        on the lists in __init__. 
        """
        
        # initializing the list
        cell_types = []
        
        # looping over all the rows
        for n in np.arange(len(self.drift_df.index)):
            cell = (self.drift_df['Side'][n], self.drift_df['Column'][n], 
            		self.drift_df['Layer'][n])  # gives location coordinates 
                    # of the cell
            
            # consults the lists to determine what type of cell it is
            if cell in self.edge_cells:
                region = 'edge'
            elif cell in self.corner_cells:
                region = 'corner'
            else:
                region = 'center'
            
            # calls define_io to determine location within the cell
            inner = self.define_io(self.drift_df['Drift_time'][n], region)  

            # adds the in/out label so the result is compatible with the 
            # parameter dictionary keys
            if inner is True:
                cell_types.append(region + '_in')
            else: 
                cell_types.append(region + '_out')
            
        self.cell_types = pd.Series(data=cell_types, name='Cell_type')

    def find_params(self, cell_type):
        """ Consults the ab_vals dictionary to determine a and b parameter 
        values.
        
        :param cell_type: Defines the cell type and location within the cell, 
        e.g. 'center_in'.
        :type cell_type: str
        """

        pressure = self.pressure
        
        if pressure == 850:
            n = 0
        elif pressure == 880:
            n = 1
        elif pressure == 910:
            n = 2 
        
        # finds the parameters stored in the dictionary based on region and 
        # pressure
        params = self.ab_vals[cell_type][n]
        a = params[0]
        b = params[1]
        tx = params[2]

        return params

    def calc_radius(self):
        """ Calculates the radius of the particle based on drift time and cell 
        type.
        """
        
        # initializing the list
        radii = []
        
        # loops over all rows
        for n in np.arange(len(self.drift_df.index)):
            params = self.find_params(self.cell_types[n])  # calls find_params 
            # to define a and b
            a = params[0]
            b = params[1]

            rad = (self.drift_times[n] / a)**(1 / (1 - b))  # calculates the 
            # radius
            radii.append(rad)
        
        self.drift_radii = pd.Series(data=radii, name='Drift_radius')