# GUEZI RAG Chatbot - Plan d'Action

## Problèmes Identifiés

1. **RAG ne fonctionne pas** - Likutei Moharan Torah 1 non trouvé
2. **TTS ne fonctionne pas** - Mauvais modèle ou configuration
3. **Pas d'entrée vocale** - Microphone non implémenté
4. **Textes hébreux** - Problème d'embedding/recherche
5. **Pas de génération d'images**
6. **Supabase non fonctionnel**
7. **Pas de repo GitHub**

## Plan d'Action (Dans l'ordre)

### Phase 1: Diagnostic RAG (CRITIQUE)
- [ ] Vérifier si Likutei Moharan 1 existe dans les données
- [ ] Vérifier si le texte est correctement chunké
- [ ] Vérifier si les embeddings sont corrects
- [ ] Tester la recherche directement

### Phase 2: Corriger le RAG
- [ ] S'assurer que TOUS les textes sont présents
- [ ] Améliorer la recherche (hébreu + anglais)
- [ ] Tester avec plusieurs requêtes

### Phase 3: Recherche Modèles Gemini
- [ ] Lister tous les modèles Gemini disponibles
- [ ] Identifier modèle TTS correct
- [ ] Identifier modèle pour Live API (voice input)
- [ ] Identifier modèle pour génération d'images

### Phase 4: Implémenter Audio
- [ ] TTS fonctionnel
- [ ] Input microphone (STT)
- [ ] Interface voice conversation

### Phase 5: Génération d'Images
- [ ] Intégrer Imagen/Gemini pour images

### Phase 6: Infrastructure
- [ ] Configurer Supabase correctement
- [ ] Créer repo GitHub
- [ ] Déployer sur Streamlit Cloud

## Vérification Finale
- [ ] Test complet RAG (toutes les sources)
- [ ] Test TTS
- [ ] Test microphone
- [ ] Test génération images
- [ ] Test multi-langue
