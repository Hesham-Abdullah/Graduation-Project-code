from data import DataSet

if __name__ == "__main__":
    
    data = DataSet(seq_length=30, class_limit=None)
    data.export_classes_to_file()