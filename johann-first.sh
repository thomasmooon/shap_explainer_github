mkdir delme
cd delme

# @Johann: please provide path (from *within* the container) to those two files
H5FULLPATH= # maybe H5FULLPATH="/root/work/data/mh1_fu1_CNN1D_subdomain.h5"
CVFULLPATH= # maybe CVFULLPATH="/root/work/data/censored.incident.cvsamp.csv"

# download + install project and shap_explainer package
git clone https://gitlab.com/tlinden/shap_explainer.git
cd shap_explainer

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