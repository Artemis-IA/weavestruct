Voici la liste complète des méthodes de la classe `MLflowClient` dans la version 2.17.2 :

### **Méthodes de gestion des expériences :**

1. **`create_experiment(name, artifact_location=None, tags=None)`**
2. **`get_experiment(experiment_id)`**
3. **`get_experiment_by_name(name)`**
4. **`list_experiments(view_type=ViewType.ACTIVE_ONLY, max_results=None, page_token=None)`**
5. **`delete_experiment(experiment_id)`**
6. **`restore_experiment(experiment_id)`**
7. **`rename_experiment(experiment_id, new_name)`**
8. **`set_experiment_tag(experiment_id, key, value)`**
9. **`delete_experiment_tag(experiment_id, key)`**

### **Méthodes de gestion des runs :**

10. **`create_run(experiment_id, start_time=None, tags=None, run_name=None)`**
11. **`get_run(run_id)`**
12. **`delete_run(run_id)`**
13. **`restore_run(run_id)`**
14. **`list_run_infos(experiment_id, run_view_type=ViewType.ACTIVE_ONLY, max_results=None, page_token=None)`**
15. **`search_runs(experiment_ids=None, filter_string='', run_view_type=ViewType.ACTIVE_ONLY, max_results=100, order_by=None, page_token=None)`**
16. **`set_terminated(run_id, status=None, end_time=None)`**
17. **`log_param(run_id, key, value)`**
18. **`log_params(run_id, params)`**
19. **`log_metric(run_id, key, value, timestamp=None, step=None)`**
20. **`log_metrics(run_id, metrics, timestamp=None, step=None)`**
21. **`set_tag(run_id, key, value)`**
22. **`set_tags(run_id, tags)`**
23. **`delete_tag(run_id, key)`**
24. **`log_batch(run_id, metrics=None, params=None, tags=None, dataset_inputs=None)`**
25. **`get_metric_history(run_id, key)`**
26. **`get_metric_history_bulk(run_id, metric_keys)`**

### **Méthodes de gestion des artefacts :**

27. **`log_artifact(run_id, local_path, artifact_path=None)`**
28. **`log_artifacts(run_id, local_dir, artifact_path=None)`**
29. **`list_artifacts(run_id, path=None)`**
30. **`download_artifacts(run_id, path, dst_path=None)`**
31. **`upload_artifact(run_id, artifact_file, artifact_path=None)`**
32. **`upload_artifacts(run_id, artifact_dir, artifact_path=None)`**
33. **`get_artifact_uri(run_id, artifact_path=None)`**
34. **`download_artifact_from_uri(artifact_uri, output_path=None)`**

### **Méthodes du registre de modèles :**

35. **`create_registered_model(name, tags=None, description=None)`**
36. **`get_registered_model(name)`**
37. **`search_registered_models(filter_string='', max_results=None, order_by=None, page_token=None)`**
38. **`delete_registered_model(name)`**
39. **`rename_registered_model(name, new_name)`**
40. **`update_registered_model(name, description=None)`**
41. **`set_registered_model_tag(name, key, value)`**
42. **`delete_registered_model_tag(name, key)`**
43. **`get_latest_versions(name, stages=None)`**
44. **`create_model_version(name, source, run_id=None, tags=None, run_link=None, description=None)`**
45. **`get_model_version(name, version)`**
46. **`delete_model_version(name, version)`**
47. **`update_model_version(name, version, description=None)`**
48. **`transition_model_version_stage(name, version, stage, archive_existing_versions=False)`**
49. **`set_model_version_tag(name, version, key, value)`**
50. **`delete_model_version_tag(name, version, key)`**
51. **`get_model_version_download_uri(name, version)`**
52. **`search_model_versions(filter_string='', max_results=None, order_by=None, page_token=None)`**

### **Méthodes de gestion des datasets :**

53. **`create_dataset(name, source=None, schema=None, profile=None, tags=None, description=None)`**
54. **`get_dataset(name)`**
55. **`delete_dataset(name)`**
56. **`search_datasets(filter_string='', max_results=None, order_by=None, page_token=None)`**
57. **`set_dataset_tag(name, key, value)`**
58. **`delete_dataset_tag(name, key)`**
59. **`create_dataset_version(dataset_name, source, schema=None, profile=None, description=None, tags=None)`**
60. **`get_dataset_version(dataset_name, version)`**
61. **`delete_dataset_version(dataset_name, version)`**
62. **`search_dataset_versions(filter_string='', max_results=None, order_by=None, page_token=None)`**
63. **`set_dataset_version_tag(dataset_name, version, key, value)`**
64. **`delete_dataset_version_tag(dataset_name, version, key)`**

### **Méthodes pour les entrées de run :**

65. **`log_inputs(run_id, datasets=None, tags=None)`**

### **Méthodes diverses :**

66. **`set_tracking_uri(uri)`**
67. **`get_tracking_uri()`**
68. **`set_registry_uri(uri)`**
69. **`get_registry_uri()`**

Cette liste couvre toutes les méthodes publiques disponibles dans la classe `MLflowClient` pour la version 2.17.2, conformément au code source disponible sur le dépôt GitHub de MLflow.