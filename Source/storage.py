import sqlite3
import faiss
import ujson
import traceback

import time

class Storage:
    def __init__(self, db_name):
        try:
            self.db_name = db_name
            self.conn = sqlite3.connect(db_name, check_same_thread=False)  # This allows the connection to be used in multiple threads
            self.cursor = self.conn.cursor()
            self.optimize_database()
        except sqlite3.Error as e:
            print(f"Failed to connect to database {db_name}: {e}")
            raise e  # Re-raise exception after logging to handle it upstream if needed
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def optimize_database(self):
        self.cursor.execute("PRAGMA cache_size = -64000;")  
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA synchronous = OFF;")

    def create_dataset(self, dataset_name):
        try:
            # Create a new table with 'id' as the primary key and 'data' for JSON storage, if it doesn't already exist.
            self.cursor.execute(f'''CREATE TABLE IF NOT EXISTS {dataset_name} (id TEXT PRIMARY KEY, data TEXT)''')
            self.conn.commit()  # Commit changes to the database.
        except sqlite3.Error as e:
            # Handle SQLite errors, e.g., syntax errors in the SQL command.
            print(dataset_name)
            print(f"Database error: {e}")
        except Exception as e:
            # Handle unexpected errors, keeping the program from crashing.
            print(f"Exception in create_dataset: {e}")

            
    def insert_dict(self, dataset_name, data_dict):
        # Extract 'id' from data dictionary.
        dict_id = data_dict.get('id')
        
        # Proceed only if 'id' exists and it's not already in the dataset.
        if dict_id is not None and not self.is_id_in_dataset(dataset_name, dict_id):
            try:
                # Convert dictionary to JSON string.
                data_json = ujson.dumps(data_dict)
                # Insert 'id' and JSON string into the specified dataset.
                self.cursor.execute(f"INSERT INTO {dataset_name} (id, data) VALUES (?, ?)", (dict_id, data_json,))
                self.conn.commit()  # Commit the transaction.
            except sqlite3.Error as e:
                # Handle SQLite errors during insert operation.
                print(f"Database error: {e}")
            except Exception as e:
                # Handle other unexpected errors.
                print(f"Exception in insert_dict: {e}")

    def insert_dict_new(self, dataset_name, document, index):
        import json
        # Insert a dictionary as a JSON string into the specified table
        try:
            data_str = json.dumps(document)
            # Check if the ID already exists in the table
            self.cursor.execute(f"SELECT id FROM {dataset_name} WHERE id = ?", (index,))
            existing_id = self.cursor.fetchone()
            if existing_id:
                print(f"ID {index} already exists in {dataset_name}. Clearing table and retrying insertion.")
                # Clear the table
                self.cursor.execute(f"DELETE FROM {dataset_name}")
                self.conn.commit()
                print(f"All existing data in {dataset_name} has been cleared.")
            # Insert the new data
            self.cursor.execute(f"INSERT INTO {dataset_name} (id, data) VALUES (?, ?)", (index, data_str))
            self.conn.commit()
            print(f"Document inserted successfully into {dataset_name} with ID {index}.")
        except Exception as e:
            print(f"Error inserting into table {dataset_name}: {e}")


    def insert_or_update_dict(self, dataset_name, data_dict):
        # Extract 'id' from data dictionary.
        dict_id = data_dict.get('id')
        
        # Proceed only if 'id' exists.
        if dict_id is not None:
            try:
                # Convert dictionary to JSON string.
                data_json = ujson.dumps(data_dict)
                # Use 'REPLACE INTO' to insert or update the row with the specified 'id'.
                self.cursor.execute(f"REPLACE INTO {dataset_name} (id, data) VALUES (?, ?)", (dict_id, data_json,))
                self.conn.commit()  # Commit the transaction.
            except sqlite3.Error as e:
                # Handle SQLite errors during insert/update operation.
                print(f"Database error: {e}")
            except Exception as e:
                # Handle other unexpected errors.
                print(f"Exception in insert_or_update_dict: {e}")
                print(traceback.format_exc())


    def is_id_in_dataset(self, dataset_name, dict_id):
        try:
            # Execute a SQL query to check if the given 'dict_id' exists in the 'dataset_name' table.
            self.cursor.execute(f"SELECT 1 FROM {dataset_name} WHERE id = ?", (dict_id,))
            # Return True if the ID exists, False otherwise.
            return self.cursor.fetchone() is not None
        except sqlite3.Error as e:
            # Handle SQLite errors, logging the issue and indicating the ID was not found.
            print(f"Database error: {e}")
            return False

    
    def store_faiss_data(self, identifier, index, data_mapping):
        try:
            # Serialize the FAISS index into bytes for storage.
            serialized_index = faiss.serialize_index(index).tobytes()
            # Convert the data mapping dictionary into a JSON string.
            json_data_mapping = ujson.dumps(data_mapping)
            
            # Ensure the storage table exists, creating it if necessary.
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS faiss_index_data 
                                (id TEXT PRIMARY KEY, 
                                    faiss_index BLOB, 
                                    data_mapping TEXT)''')

            # Insert the serialized index and JSON data mapping into the database.
            self.cursor.execute("INSERT INTO faiss_index_data (id, faiss_index, data_mapping) VALUES (?, ?, ?)",
                                (identifier, serialized_index, json_data_mapping,))
            self.conn.commit()  # Commit the changes to the database.
        except sqlite3.Error as e:
            # Log database-related errors.
            print(f"Database error: {e}")
        except Exception as e:
            # Log any other exceptions.
            print(f"Exception in store_faiss_data: {e}")


    def retrieve_faiss_data(self, identifier):
        try:
            # Correct the SQL query to fetch the faiss_index and data_mapping for the given identifier.
            # Remove the parentheses around the selected columns to ensure proper data retrieval.
            self.cursor.execute("SELECT faiss_index, data_mapping FROM faiss_index_data WHERE id = ?", (identifier,))
            row = self.cursor.fetchone()

            if row is not None:
                # Correctly deserialize the FAISS index from the binary data stored in the database.
                # Use faiss.deserialize_index directly on the binary data without converting it back to an array.
                index = faiss.deserialize_index(row[0])

                # Deserialize the JSON string back into a Python dictionary.
                data_mapping = ujson.loads(row[1])

                return index, data_mapping
            else:
                # Return None if no entry was found for the given identifier.
                return None, None
        except sqlite3.Error as e:
            # Log SQLite errors and return None to indicate failure.
            print(f"Database error: {e}")
            return None, None
        except Exception as e:
            # Log unexpected errors and return None to indicate failure.
            print(f"Exception in retrieve_faiss_data: {e}")

    def retrieve_dicts(self, dataset_name):
        # Properly quote the table name to prevent SQL injection and syntax errors.
        safe_dataset_name = f'"{dataset_name}"'
        start= time.time()
        # Execute a SQL query to fetch all records from the specified dataset table.
        self.cursor.execute(f"SELECT * FROM {safe_dataset_name}")
        rows = self.cursor.fetchall()
        end=time.time()
        print(f"-------------------------{end-start}")
        # Utilizing a faster JSON parsing library if significant JSON parsing overhead is detected
        start= time.time()
        a= [ujson.loads(row[1]) for row in rows]
        end=time.time()
        print(f"UJSON-------------------------{end-start}")

        return a

    def reset_database(self):
        try:
            # Retrieve the names of all tables in the database.
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = self.cursor.fetchall()

            # Iterate over each table name and drop the table to remove it from the database.
            for table in tables:
                self.cursor.execute(f"DROP TABLE {table[0]}")
            
            # Commit the changes to finalize the removal of all tables.
            self.conn.commit()
        except sqlite3.Error as e:
            # Handle and log any SQLite errors encountered during the operation.
            print(f"Database error: {e}")
        except Exception as e:
            # Handle and log any non-SQLite errors that may occur.
            print(f"Exception in reset_database: {e}")

    def get_dict_by_id(self, dataset_name, dict_id):
        try:
            # Execute SQL query to fetch the record with the specified ID from the given dataset.
            self.cursor.execute(f"SELECT data FROM {dataset_name} WHERE id = ?", (dict_id,))
            row = self.cursor.fetchone()
            
            # If the record exists, convert the JSON string in 'data' column back to a dictionary and return it.
            if row is not None:
                return ujson.loads(row[0])
            else:
                # Return None if no record was found with the given ID.
                return None
        except sqlite3.Error as e:
            # Log database-related errors and return None to indicate failure.
            print(f"Database error: {e}")
            Storage.get_dict_by_id(self, dataset_name, dict_id)
            
        except Exception as e:
            # Log any other exceptions that occur and return None to indicate failure.
            print(f"Exception in get_dict_by_id: {e}")

    def close(self):
        try:
            # Attempt to close the database connection.
            self.conn.close()
        except sqlite3.Error as e:
            # Log any SQLite errors encountered during the closing process.
            print(f"Database error: {e}")
        except Exception as e:
            # Log any other exceptions that might occur during closure.
            print(f"Exception in close: {e}")
            print(traceback.format_exc())

