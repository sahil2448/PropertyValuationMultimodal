from sentinelhub import SHConfig

config = SHConfig()
config.sh_client_id = SENTINEL_CLIENT_ID
config.sh_client_secret = SENTINEL_CLIENT_SECRET

config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_auth_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

config.save()

print("✓ Configuration saved!")
print(f"Base URL: {config.sh_base_url}")
