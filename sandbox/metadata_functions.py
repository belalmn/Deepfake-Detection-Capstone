import json
import pandas as pd

class Metadata:
    def __init__(self, file_path, sample=True):
        """
        Metadata class

        Parameters
        ----------
        file_path : str or dict
            Path to json file or dictionary with paths to json files
        """

        # Load metadata
        if type(file_path) is dict:
            self._load_metadata(file_path, multiple=True, sample=sample)
        else:
            self._load_metadata(file_path, multiple=False, sample=sample)

    ## Public methods
    # Get metadata
    def get_metadata(self):
        """
        Get metadata

        Parameters
        ----------
        None

        Returns
        ----------
        df : pandas.DataFrame
            Metadata
        """
        return self.df
        
    ## Private methods
    # Load metadata from json file
    def _load_metadata(self, file_path, multiple=False, sample=True):
        """
        Load metadata from json file
        
        Parameters
        ----------
        file_path : str
            Path to json file
            
        Returns
        ----------
        None
        """
        # Load metadata
        with open(file_path) as json_file:
            data = json.load(json_file)
        data = pd.DataFrame.from_dict(data, orient='index')
        data.reset_index(inplace=True)

        # Rename columns
        data.rename(columns={'index':'id'}, inplace=True)
        if sample:
            data.drop(columns=['split'], inplace=True)

        # Split into original and fake datasets
        fake_df = data[data['label'] == "FAKE"]
        original_df = data[data['label'] == "REAL"]

        # Clean datasets
        fake_df.drop(columns=['label'], inplace=True)
        original_df.drop(columns=['label', 'original'], inplace=True)

        self.fake_df = fake_df
        self.original_df = original_df