pipeline {
    agent any

    environment {
        PYTHON    = "python3"
        VENV_DIR  = ".venv"
        OUTPUT_DIR = "artifacts"
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python Env') {
            steps {
                sh """
                # Create venv only if it doesn't exist
                if [ ! -d "${VENV_DIR}" ]; then
                  ${PYTHON} -m venv ${VENV_DIR}
                  . ${VENV_DIR}/bin/activate
                  pip install --upgrade pip
                  # CPU-only PyTorch wheels (much smaller than CUDA)
                  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
                  pip install -r requirements.txt
                else
                  . ${VENV_DIR}/bin/activate
                  pip install -r requirements.txt
                fi
                """
            }
        }

        stage('Train Model') {
            steps {
                sh """
                . ${VENV_DIR}/bin/activate
                ${PYTHON} train.py --output_dir ${OUTPUT_DIR}
                """
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: "${OUTPUT_DIR}/**", fingerprint: true
            }
        }
    }

    post {
        success {
            echo "✅ Training completed successfully."
        }
        failure {
            echo "❌ Training failed. Check console output."
        }
    }
}
