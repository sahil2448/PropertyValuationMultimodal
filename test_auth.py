from sentinelhub import SHConfig

config = SHConfig()

# Replace these with your actual credentials
config.sh_client_id = "sh-0b685b0d-19b6-4988-b298-1b0466f73b2a"
config.sh_client_secret = "97SzVcjeA8sGbb7kyeWmZcaWDSSmpiXF"
config.sh_base_url = 'https://sh.dataspace.copernicus.eu'
config.sh_token_url = 'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'

try:
    from sentinelhub import SentinelHubSession
    session = SentinelHubSession(config)
    token = session.token
    print("✓ Authentication successful!")
    print(f"Token (first 50 chars): {token[:50]}...")
except Exception as e:
    print(f"✗ Authentication failed: {e}")
