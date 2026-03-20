import redis
import pandas as pd
from tqdm.asyncio import tqdm

class RedisCRUDHandler:
    def __init__(self, r : redis.Redis):
        self.r = r

    def create(self, file_path):

        try:
            self.r.ping()
        except Exception as e:
            print(f"Error al conectar con Redis: {e}")
            return 0

        df = pd.read_csv(file_path)
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Cargando datos en Redis"):
            key = f"cards:{row['code']}"
            self.r.hset(key, mapping=row.to_dict())
        return 1

    def is_in(self, code : str) -> bool:
        key = f"cards:{code}"
        return self.r.exists(key) == 1

    def read(self, code : str) -> dict:
        key = f"cards:{code}"
        if self.r.exists(key) == 0:
            print(f"La carta con código {code} no existe en Redis.")
            return {}
        return self.r.hgetall(key)

    def delete(self, code : str) -> bool:
        key = f"cards:{code}"
        if self.r.exists(key) == 0:
            print(f"La carta con código {code} no existe en Redis.")
            return False
        self.r.delete(key)
        return True