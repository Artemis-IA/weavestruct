from locust import HttpUser, TaskSet, task

class UserBehavior(TaskSet):
    @task
    def index(self):
        self.client.get("/")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    min_wait = 5000
    max_wait = 9000
