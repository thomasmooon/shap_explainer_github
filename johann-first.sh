rm -rf delme
SHAPDIR=delme
mkdir $SHAPDIR
cd $SHAPDIR

# download + install project and shap_explainer package
git clone https://github.com/thomasmooon/shap_explainer_github.git
cd shap_explainer_github

# copy files
H5FILE=/root/work/data/mh1_fu1_CNN1D_subdomain.h5
CVFILE=/root/data/censored.incident.cvsamp.csv
cp $H5FILE data/processed/
cp $CVFILE data/processed/

# copy keras disease models
SOURCE=/root/src/AWS_finalmodel/q30_train_finalmodel/results/*.h5
TARGET=/root/$SHAPDIR/models/keras
mv $SOURCE $TARGET

