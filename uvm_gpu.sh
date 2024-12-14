# Installer le module si nécessaire
# sudo apt install nvidia-modprobe

#----------------------------------
# Charge dynamiquement les modules NVIDIA pour permettre l'accès au GPU par des applications non root
# Décharge le module NVIDIA Unified Memory Manager du noyau
# Recharge le module NVIDIA Unified Memory Manager dans le noyau
sudo nvidia-modprobe -u
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

#----------------------------------
# (Optionnel) Redémarrer le serveur Ollama  via le gestionnaire de services systemd
sudo systemctl restart ollama

# ----------------------------------
# (Divers) Supprimer le "verouillage" CodeCarbon de la session
# rm /tmp/.codecarbon.lock
