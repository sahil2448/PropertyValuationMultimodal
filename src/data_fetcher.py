import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SentinelHubDownloadClient,
    DownloadRequest
)

from config import (
    SENTINEL_CLIENT_ID,
    SENTINEL_CLIENT_SECRET,
    IMAGE_SIZE,
    IMAGES_DIR,
    RAW_DATA_DIR
)

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"


class SatelliteImageFetcher:
    def __init__(self):
        if not SENTINEL_CLIENT_ID or not SENTINEL_CLIENT_SECRET:
            raise ValueError("Missing SENTINEL_CLIENT_ID / SENTINEL_CLIENT_SECRET in .env")

        self.config = SHConfig()
        self.config.sh_client_id = SENTINEL_CLIENT_ID
        self.config.sh_client_secret = SENTINEL_CLIENT_SECRET
        self.config.sh_token_url = CDSE_TOKEN_URL 

        self.client = SentinelHubDownloadClient(config=self.config)

        self.buffer_m = 300

        self.max_workers = 6

        print("✓ Sentinel Hub client ready (CDSE)")

    def _bbox_from_point(self, lat: float, lon: float, buffer_m: float) -> BBox:
        dlat = buffer_m / 111_000.0
        dlon = buffer_m / 75_000.0  
        return BBox([lon - dlon, lat - dlat, lon + dlon, lat + dlat], crs=CRS.WGS84)

    def _build_process_payload(self, lat: float, lon: float) -> dict:
        bbox = self._bbox_from_point(lat, lon, self.buffer_m)

        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: ["B04","B03","B02"],
            output: { bands: 3 }
          };
        }
        function evaluatePixel(s) {
          return [2.5*s.B04, 2.5*s.B03, 2.5*s.B02];
        }
        """

        req = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=("2023-01-01", "2024-12-31"),
                    maxcc=0.2
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=bbox,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            config=self.config
        )

        try:
            return req.get_request_payload()
        except AttributeError:
            for candidate in ("get_payload", "get_request_body", "payload"):
                if hasattr(req, candidate):
                    attr = getattr(req, candidate)
                    return attr() if callable(attr) else attr

            print("Available methods/attrs on SentinelHubRequest:", [m for m in dir(req) if not m.startswith("__")])
            raise RuntimeError("Could not obtain request payload from SentinelHubRequest; inspect available methods above.")

    # def download_one(self, prop_id, lat, lon, save_path: str, retries: int = 3) -> str | None:
    #     if os.path.exists(save_path):
    #         return save_path

    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)

    #     payload = self._build_process_payload(lat, lon)

    #     dl_req = DownloadRequest(
    #         request_type="POST",
    #         url=CDSE_PROCESS_URL,
    #         post_values=payload,
    #         data_type=MimeType.PNG,
    #         headers={"content-type": "application/json"},
    #         use_session=True
    #     )

    #     backoff = 1.0
    #     for attempt in range(retries):
    #         try:
    #             download_result = self.client.download([dl_req], decode_data=True)

    #             if isinstance(download_result, (list, tuple)):
    #                 img_arr = download_result[0]
    #             else:
    #                 img_arr = download_result

    #             if hasattr(img_arr, "dtype") and img_arr.dtype != np.uint8:
    #                 img_arr = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)

    #             Image.fromarray(img_arr).save(save_path, "JPEG", quality=90)
    #             return save_path

    #         except Exception as e:
    #             if attempt == retries - 1:
    #                 print(f"[FAIL] id={prop_id} lat={lat} lon={lon} err={e}")
    #                 return None
    #             time.sleep(backoff + (backoff * 0.25 * np.random.rand()))
    #             backoff *= 2.0

    #     return None

    # def fetch_images_for_dataset(self, dataset_filename: str, dataset_type: str):
    #     df = pd.read_csv(RAW_DATA_DIR / dataset_filename)

    #     save_dir = IMAGES_DIR / dataset_type
    #     save_dir.mkdir(parents=True, exist_ok=True)

    #     if "id" not in df.columns:
    #         df["id"] = range(len(df))

    #     tasks = []
    #     results = {}

    #     for idx, row in df.iterrows():
    #         prop_id = row["id"]
    #         lat = float(row["lat"])
    #         lon = float(row["long"])
    #         save_path = str(save_dir / f"property_{prop_id}.jpg")
    #         tasks.append((idx, prop_id, lat, lon, save_path))

    #     with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
    #         future_map = {
    #             ex.submit(self.download_one, prop_id, lat, lon, save_path): idx
    #             for (idx, prop_id, lat, lon, save_path) in tasks
    #         }

    #         for fut in tqdm(as_completed(future_map), total=len(future_map), desc=f"{dataset_type} images"):
    #             idx = future_map[fut]
    #             try:
    #                 results[idx] = fut.result()
    #             except Exception as e:
    #                 results[idx] = None
    #                 print(f"[THREAD-ERR] idx={idx} err={e}")


    #     df["image_path"] = [results.get(i) for i in range(len(df))]

    #     out_csv = RAW_DATA_DIR / f"{dataset_type}_with_images.csv"
    #     df.to_csv(out_csv, index=False)

    #     ok = df["image_path"].notna().sum()
    #     print(f"Done {dataset_type}: {ok}/{len(df)} saved -> {out_csv}")


    def download_one(self, prop_id, lat, lon, save_path: str, retries: int = 3) -> str | None:
        """
        Saves to a temp file first (atomic rename). Returns final path or None on fail.
        """
        if os.path.exists(save_path):
            return save_path

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        temp_path = f"{save_path}.part"

        payload = self._build_process_payload(lat, lon)

        dl_req = DownloadRequest(
            request_type="POST",
            url=CDSE_PROCESS_URL,
            post_values=payload,
            data_type=MimeType.PNG,
            headers={"content-type": "application/json"},
            use_session=True
        )

        backoff = 1.0
        for attempt in range(retries):
            try:
                download_result = self.client.download([dl_req], decode_data=True)
                img_arr = download_result[0] if isinstance(download_result, (list, tuple)) else download_result

                if hasattr(img_arr, "dtype") and img_arr.dtype != np.uint8:
                    img_arr = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)

                Image.fromarray(img_arr).save(temp_path, "JPEG", quality=90)
                os.replace(temp_path, save_path)
                return save_path

            except Exception as e:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                if attempt == retries - 1:
                    print(f"[FAIL] id={prop_id} lat={lat} lon={lon} err={e}")
                    return None
                time.sleep(backoff + 0.25 * np.random.rand())
                backoff *= 2.0

        return None


    def fetch_images_for_dataset(self, dataset_filename: str, dataset_type: str, flush_every: int = 200):
        """
        Resumable fetch:
         - If <dataset_type>_with_images.csv exists, load it and skip rows with valid image paths.
         - Periodically flush progress to disk (every flush_every saves).
        """
        in_csv = RAW_DATA_DIR / dataset_filename
        out_csv = RAW_DATA_DIR / f"{dataset_type}_with_images.csv"

        df = pd.read_csv(in_csv)

        if "id" not in df.columns:
            df["id"] = range(len(df))

        if out_csv.exists():
            existing = pd.read_csv(out_csv)
            existing = existing.set_index("id")
            df = df.set_index("id")
            if "image_path" in existing.columns:
                df["image_path"] = existing["image_path"]
            df = df.reset_index()
        else:
            df["image_path"] = None

        save_dir = IMAGES_DIR / dataset_type
        save_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for idx, row in df.iterrows():
            prop_id = row["id"]
            lat = float(row["lat"])
            lon = float(row["long"])
            save_path = str(save_dir / f"property_{prop_id}.jpg")

            if pd.notna(row.get("image_path")) and os.path.exists(row["image_path"]):
                continue
            if os.path.exists(save_path):
                df.at[idx, "image_path"] = save_path
                continue

            tasks.append((idx, prop_id, lat, lon, save_path))

        if not tasks:
            print(f"No missing images for {dataset_type}. Output: {out_csv}")
            df.to_csv(out_csv, index=False)
            return

        results = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_map = {
                ex.submit(self.download_one, prop_id, lat, lon, save_path): (idx, prop_id, save_path)
                for (idx, prop_id, lat, lon, save_path) in tasks
            }

            for fut in tqdm(as_completed(future_map), total=len(future_map), desc=f"{dataset_type} images"):
                idx, prop_id, save_path = future_map[fut]
                try:
                    res = fut.result()
                    results[idx] = res
                    if res:
                        df.at[idx, "image_path"] = res
                except Exception as e:
                    results[idx] = None
                    df.at[idx, "image_path"] = None
                    print(f"[THREAD-ERR] idx={idx} id={prop_id} err={e}")

                completed += 1
                if completed % flush_every == 0:
                    df.to_csv(out_csv, index=False)
                    print(f"[FLUSH] wrote progress after {completed} completed (so far).")

        df.to_csv(out_csv, index=False)
        ok = df["image_path"].notna().sum()
        print(f"Done {dataset_type}: {ok}/{len(df)} saved -> {out_csv}")

def main():
    fetcher = SatelliteImageFetcher()
    fetcher.fetch_images_for_dataset("train1.csv", "train")
    fetcher.fetch_images_for_dataset("test2.csv", "test")


if __name__ == "__main__":
    main()
