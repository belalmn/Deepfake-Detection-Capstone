import json
import pandas as pd

class Metadata:
    """
    Metadata class

    Attributes
    ----------
    df : pandas.DataFrame
        Metadata

    Public Methods
    ----------
    metadata()
        Get metadata
    """

    def __init__(self, file_path, sample=True):
        """
        Metadata class

        Parameters
        ----------
        file_path : str or dict
            Path to json file or dictionary with paths to json files
        """
        self.sample = sample
        # Load metadata
        if type(file_path) is dict:
            self._load_metadata(file_path, multiple=True, sample=sample)
        else:
            self._load_metadata(file_path, multiple=False, sample=sample)

    ## Public methods
    # Get metadata
    def original(self):
        """
        Get original metadata
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        pandas.DataFrame
            Original metadata
        """
        return self.original_df
    
    def fake(self):
        """
        Get fake metadata
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        pandas.DataFrame
            Fake metadata
        """
        return self.fake_df
    
    def is_sample(self):
        """
        Check if metadata is a sample
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        bool
            True if metadata is a sample, False otherwise
        """
        return self.sample
        
    def get_sample_pairs(self):
        """
        Get sample pairs

        Parameters
        ----------
        None

        Returns
        ----------
        pandas.DataFrame
            Sample pairs
        """
        df = pd.merge(self.original_df, self.fake_df, left_on='id', right_on='original', how='inner', suffixes=('_original', '_fake')).drop(columns=['original'])
        return df
    
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
        data = None
        if multiple:
            file_paths = file_path
            for folder, file_path in file_paths.items():
                with open(file_path) as json_file:
                    new_data = json.load(json_file)
                new_data = pd.DataFrame.from_dict(data, orient='index')
                new_data.reset_index(inplace=True)
                data['folder'] = folder
                data = pd.concat([data, new_data])
        else:
            with open(file_path) as json_file:
                data = json.load(json_file)
            data = pd.DataFrame.from_dict(data, orient='index')
            data.reset_index(inplace=True)

        # Rename columns
        data.rename(columns={'index':'id'}, inplace=True)
        data.drop(columns=['split'], inplace=True)

        # Remove .mp4 from all ids
        data['id'] = data['id'].apply(lambda x: x[:-4])
        
        # Split into original and fake datasets
        fake_df = data[data['label'] == "FAKE"]
        original_df = data[data['label'] == "REAL"]

        # Clean datasets
        fake_df.drop(columns=['label'], inplace=True)
        fake_df['original'] = fake_df['original'].apply(lambda x: x[:-4])
        original_df.drop(columns=['label', 'original'], inplace=True)

        self.fake_df = fake_df
        self.original_df = original_df