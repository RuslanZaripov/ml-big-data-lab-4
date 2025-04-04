# ml-big-data-lab-4

Лабортаорная № 4. ИНТЕГРАЦИЯ APACHE KAFKA СЕРВИСА

Цель: Получить навыки реализации Kafka Producer и Consumer и их последующей интеграции.

Стек: Python 3.10, Jupyter, FastApi, redis, scikit-learn, DVC, Docker, Jenkins, Ansible Vault, Kafka

[Письменный отчет](static/ИБД-Лаб-4.pdf)

## Ansible Vault

```python
pip install ansible-vault-win
```

## Encrypt and decrypt

- write password into `env/.vault-pass`

- write configuration parameters into `env/secrets.env`

```bash
ansible-vault encrypt env/secrets.env --vault-pass-file env/.vault-pass
```

```bash
ansible-vault decrypt env/secrets.env --vault-pass-file env/.vault-pass
```

## Project Structure

```text
.
├── CD
│   └── Jenkinsfile
├── CI
│   └── Jenkinsfile
├── Dockerfile
├── README.md
├── config.ini
├── data
│   └── fashion-mnist.zip
├── data.dvc
├── docker-compose.yml
├── env
│   ├── secrets.env
│   └── vault-pass.env
├── experiments
├── notebooks
├── requirements.txt
├── src
│   ├── app.py
│   ├── database.py
│   ├── kafka_consumer.py
│   ├── kafka_producer.py
│   ├── logger.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   └── unit_tests
│       ├── __init__.py
│       ├── test_app.py
│       ├── test_database.py
│       ├── test_preprocess.py
│       └── test_training.py
├── static
└── tests
    ├── test_0.json
    └── test_1.json
```

## Docker Hub

- web service image available [here](https://hub.docker.com/repository/docker/zarus03/ml-big-data-lab-4)
