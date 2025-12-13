# Initialize repo
git init MedicalCostProject
cd MedicalCostProject

# Add remote
git remote add origin https://github.com/MelihGulum/Comprehensive-Data-Science-AI-Project-Portfolio.git

# Enable sparse checkout
git sparse-checkout init --cone

# Set the folder you want (handle spaces)
git sparse-checkout set "Machine Learning Projects/03. Medical Cost Prediction"

# Pull only that folder
git pull origin main
