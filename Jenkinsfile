pipeline{
    agent any

    stages{
        stage("Checkout"){
            steps{
                checkout([$class: 'GitSCM', branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'ef70b0d3-000a-4c42-a98f-d2c045e33251', url: 'https://github.com/Jyothsna9999/bank-term-deposit-prediction.git']]])
            }
        }
        stage("build"){
            steps{
                git branch: 'main', credentialsId: 'ef70b0d3-000a-4c42-a98f-d2c045e33251', url: 'https://github.com/Jyothsna9999/bank-term-deposit-prediction.git'
            }
        }
        stage("load_data"){
            steps{
                sh 'python3 load_data.py'
            }
        }
        stage("data_analysis"){
            steps{
                sh 'python3 data_analysis.py'
            }
        }
        stage("feature_engineering"){
            steps{
                sh 'python3 feature_engineering.py'
            }
        }
        stage("data_preprocess"){
            steps{
                sh 'python3 data_proprocess.py'
            }
        }
        stage("model_selection"){
            steps{
                sh 'python3 model_selection.py'
            }
        }
        
    }
    post{
        always{
            emailext body:"summary", subject: "Pipeline Status", to: 'Jyothsna.Pendyala@cloudangles.com'
        }
    }
}