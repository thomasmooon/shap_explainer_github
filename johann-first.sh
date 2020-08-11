mkdir delme
cd delme


# --- @Johann need edit: start 
# @Johann: please provide path (from *within* the container) to those two files
# find . -name "mh1_fu1_CNN1D_subdomain.h5"
# find . -name "censored.incident.cvsamp.csv"
H5FULLPATH=$(find . -name "mh1_fu1_CNN1D_subdomain.h5")
CVFULLPATH=$(find . -name "censored.incident.cvsamp.csv")

# example 
#H5FULLPATH=E610046/TGDL-data/data/mh1_fu1_CNN1D_subdomain.h5 # 
#CVFULLPATH=E610046/TGDL-data/data/censored.incident.cvsamp.csv # 
# --- @Johann need edit: stop

# download + install project and shap_explainer package
git clone https://github.com/thomasmooon/shap_explainer_github.git
cd shap_explainer_github

# copy files
cp $H5FULLPATH data/processed/
cp $CVFULLPATH data/processed/

# download keras disease models from gdrive
cd models/keras
yes | pip install gdown
gdown https://drive.google.com/uc?id=1U3lhjaU6rWJpWYP6K8Fa8iCxzpsDLwHY
tar zxvf cnn_finalmodels.tgz
mv results/*.h5 .
rm -rf results
cd ..
cd ..

# for test1
yes | pip install xgboost