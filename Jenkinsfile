pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'aadarshsorab/ecg-devops-app:latest'
    }

    stages {

        stage('Checkout') {
            steps {
                echo 'Checking out ECG project from GitHub...'
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                echo 'Installing Python dependencies...'

                sh '''
                    python3 -m venv jenkins-venv
                    ./jenkins-venv/bin/pip install --upgrade pip
                    ./jenkins-venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running automated tests...'

                sh '''
                    if [ -f test_app.py ]; then
                        ./jenkins-venv/bin/pytest
                    else
                        echo "No test_app.py found. Skipping tests."
                    fi
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building ECG Docker image...'

                sh '''
                    docker build -t $DOCKER_IMAGE .
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                echo 'Logging in to Docker Hub and pushing image...'

                withCredentials([
                    usernamePassword(
                        credentialsId: 'dockerhub-credentials',
                        usernameVariable: 'DOCKER_USERNAME',
                        passwordVariable: 'DOCKER_PASSWORD'
                    )
                ]) {
                    sh '''
                        echo "$DOCKER_PASSWORD" | docker login \
                            --username "$DOCKER_USERNAME" \
                            --password-stdin

                        docker push $DOCKER_IMAGE
                    '''
                }
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo 'Deploying ECG application to Kubernetes...'

                sh '''
                    kubectl apply -f deployment.yaml --validate=false
                    kubectl apply -f service.yaml --validate=false
                '''
            }
        }

        stage('Verify Deployment') {
            steps {
                echo 'Checking Kubernetes deployment...'

                sh '''
                    kubectl get deployments
                    kubectl get pods
                    kubectl get services
                '''
            }
        }
    }

    post {
        success {
            echo 'ECG CI/CD Pipeline completed successfully!'
        }

        failure {
            echo 'ECG CI/CD Pipeline failed.'
        }
    }
}
```
