from glob import glob
from fastapi import FastAPI

app = FastAPI()

print("Starting services...")

modules = [module.split('\\')[1:] for module in glob('./core/servers/*/*_server.py')]
available_servers = []

for path in modules:
    exec(f'from core.servers.{path[0]}.{path[1][:-3]} import app as {path[0]}_app')
    available_servers.append(eval(path[0]+'_app'))

for server in available_servers:
    app.include_router(server.router)

print(f'Running services: {"    ".join([m[0] for m in modules])}')

