# create conda env
create conda env
conda create --name shap python=3.6.1
conda activate shap

# download + install project and shap_explainer package
yes | pip install -r requirements_CondaYes.txt
yes | pip install . # installs shap_explainer from the current directory

# adaptions for conda envs
pip uninstall --yes numpy
conda install --yes numpy
python -m pip install --upgrade scikit-image

