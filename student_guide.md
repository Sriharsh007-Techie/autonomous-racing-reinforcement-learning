# Install the latest version of UV 
- https://docs.astral.sh/uv/#installation
- make sure that you install the version for the operating system that you are using

## Open a terminal in the folder that contains all the downloaded files

## Create the uv environment
```uv venv venv_student -p 3.11.11```

## Activate the environment with
```source venv_student/bin/activate```

(for windows powershell users) 
```.\venv_student\Scripts\Activate.ps1```

## New step for bpa3
This task requires the installation of packages for pycairo.

Windows users can skip this step.

Ubuntu/Debian: ```sudo apt install libcairo2-dev pkg-config python3-dev```

macOS/Homebrew: ```brew install cairo pkg-config```

For others and questions look into https://pycairo.readthedocs.io/en/latest/getting_started.html.

##  Install the required packages with
```uv pip sync ./requirements.txt```

Note: If difficulties arise during the installation process, please consult the video tutorial for installing the RLLBC Library, as the procedure is largely analogous.
