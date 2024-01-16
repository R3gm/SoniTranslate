
language_data = {
  "english": {
    "description": """
        ### 🎥 **Translate videos easily with SoniTranslate!** 📽️

        Upload a video, audio file or provide a YouTube link. 📽️ **Gets the updated notebook from the official repository.: [SoniTranslate](https://github.com/R3gm/SoniTranslate)!**

        See the tab `Help` for instructions on how to use it. Let's start having fun with video translation! 🚀🎉
        """,
    "tutorial" : """
        # 🔰 **Instructions for use:**

        1. 📤 Upload a **video**, **audio file** or provide a 🌐 **YouTube link.**

        2. 🌍 Choose the language in which you want to **translate the video**.

        3. 🗣️ Specify the **number of people speaking** in the video and **assign each one a text-to-speech voice** suitable for the translation language.

        4. 🚀 Press the '**Translate**' button to obtain the results.

        ---

        # 🧩 **SoniTranslate supports different TTS (Text-to-Speech) engines, which are:**
        - EDGE-TTS → format `en-AU-WilliamNeural-Male` → Fast and accurate.
        - FACEBOOK MMS → format `en-facebook-mms VITS` → The voice is more natural; at the moment, it only uses CPU.
        - PIPER TTS → format `en_US-lessac-high VITS-onnx` → Same as the previous one, but it is optimized for both CPU and GPU.
        - BARK → format `en_speaker_0-Male BARK` → Good quality but slow, and it is prone to hallucinations.
        - Coqui XTTS → format `_XTTS_/AUTOMATIC.wav` → Only available for Chinese (Simplified), English, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Spanish, Hungarian, Korean and Japanese.

        ---

        # 🎤 How to Use R.V.C. and R.V.C.2 Voices (Optional) 🎶

        The goal is to apply a R.V.C. to the generated TTS (Text-to-Speech) 🎙️

        1. In the `Custom Voice R.V.C.` tab, download the models you need 📥 You can use links from Hugging Face and Google Drive in formats like zip, pth, or index. You can also download complete HF space repositories, but this option is not very stable 😕

        2. Now, go to `Replace voice: TTS to R.V.C.` and check the `enable` box ✅ After this, you can choose the models you want to apply to each TTS speaker 👩‍🦰👨‍🦱👩‍🦳👨‍🦲

        3. Adjust the F0 method that will be applied to all R.V.C. 🎛️

        4. Press `APPLY CONFIGURATION` to apply the changes you made 🔄

        5. Go back to the video translation tab and click on 'Translate' ▶️ Now, the translation will be done applying the R.V.C. 🗣️

        Tip: You can use `Test R.V.C.` to experiment and find the best TTS or configurations to apply to the R.V.C. 🧪🔍
        
        ---
        
        """,
    "tab_translate" : "Video translation",
    "video_source": "Choose Video Source",
    "link_info": "Example: www.youtube.com/watch?v=g_9rPvbENUw",
    "link_ph" : "URL goes here...",
    "dir_info": "Example: /usr/home/my_video.mp4",
    "dir_ph" : "Path goes here...",
    "sl_label" : "Source language",
    "sl_info": "This is the original language of the video",
    "tat_label" : "Translate audio to",
    "tat_info": "Select the target language and also make sure to choose the corresponding TTS for that language.",
    "num_speakers" : "Select how many people are speaking in the video.",
    "min_sk" : "Min speakers",
    "max_sk" : "Max speakers",
    "tts_select" : "Select the voice you want for each speaker.",
    "sk1" : "TTS Speaker 1",
    "sk2" : "TTS Speaker 2",
    "sk3" : "TTS Speaker 3",
    "sk4" : "TTS Speaker 4",
    "sk5" : "TTS Speaker 5",
    "sk6" : "TTS Speaker 6",
    "vc_title" : "Voice Imitation in Different Languages",
    "vc_subtitle" : """
        ### Replicate a person's voice across various languages.
        While effective with most voices when used appropriately, it may not achieve perfection in every case.
        Voice Imitation solely replicates the reference speaker's tone, excluding accent and emotion, which are governed by the base speaker TTS model and not replicated by the converter.
        This will take audio samples from the main audio for each speaker and process them.
        """,
    "vc_active_label" : "Active Voice Imitation",
    "vc_active_info" : "Active Voice Imitation: Replicates the original speaker's tone",
    "vc_segments_label" : "Max samples",
    "vc_segments_info" : "Max samples: Is the number of audio samples that will be generated for the process, more is better but it can add noise",
    "vc_dereverb_label" : "Dereverb",
    "vc_dereverb_info" : "Dereverb: Applies vocal dereverb to the audio samples.",
    "vc_remove_label" : "Remove previous samples",
    "vc_remove_info" : "Remove previous samples: Remove the previous samples generated, so new ones need to be created.",
    "xtts_title" : "Create a TTS based on an audio",
    "xtts_subtitle" : "Upload an audio file of maximum 10 seconds with a voice. Using XTTS, a new TTS will be created with a voice similar to the provided audio file.",
    "xtts_file_label" : "Upload a short audio with the voice",
    "xtts_name_label" : "Name for the TTS",
    "xtts_name_info" : "Use a simple name",
    "xtts_dereverb_label" : "Dereverb audio",
    "xtts_dereverb_info" : "Dereverb audio: Applies vocal dereverb to the audio",
    "xtts_button" : "Process the audio and include it in the TTS selector",
    "xtts_footer" : "Generate voice xtts automatically: You can use '_XTTS_/AUTOMATIC.wav' in the TTS selector to automatically generate segments for each speaker when generating the translation.",
    "extra_setting" : "Advanced Settings",
    "acc_max_label" : "Max Audio acceleration",
    "acc_max_info" : "Maximum acceleration for translated audio segments to avoid overlapping. A value of 1.0 represents no acceleration",
    "aud_mix_label" : "Audio Mixing Method",
    "aud_mix_info" : "Mix original and translated audio files to create a customized, balanced output with two available mixing modes.",
    "vol_ori" : "Volume original audio",
    "vol_tra" : "Volume translated audio",
    "sub_type" : "Subtitle type",
    "whisper_title" : "Default configuration of Whisper.",
    "srt_file_label" : "Upload an SRT subtitle file (will be used instead of the transcription of Whisper)",
    "voiceless_tk_label" : "Voiceless Track",
    "voiceless_tk_info" : "Voiceless Track: Remove the original audio voices before combining it with the translated audio. (Experimental)",
    "out_name_label" : "File name",
    "out_name_info" : "The name of the output file",
    "preview_info" : "Preview cuts the video to only 10 seconds for testing purposes. Please deactivate it to retrieve the full video duration.",
    "edit_sub_label" : "Edit generated subtitles",
    "edit_sub_info" : "Edit generated subtitles: Allows you to run the translation in 2 steps. First with the 'GET SUBTITLES AND EDIT' button, you get the subtitles to edit them, and then with the 'TRANSLATE' button, you can generate the video",
    "button_subs" : "GET SUBTITLES AND EDIT",
    "editor_sub_label" : "Generated subtitles",
    "editor_sub_info" : "Feel free to edit the text in the generated subtitles here. You can make changes to the interface options before clicking the 'TRANSLATE' button, except for 'Source language', 'Translate audio to', and 'Max speakers', to avoid errors. Once you're finished, click the 'TRANSLATE' button.",
    "editor_sub_ph" : "First press 'GET SUBTITLES AND EDIT' to get the subtitles",
    "button_translate" : "TRANSLATE",
    "output_result_label" : "DOWNLOAD TRANSLATED VIDEO",
    "sub_ori" : "Subtitles",
    "sub_tra" : "Translated subtitles",
    "ht_token_info" : "One important step is to accept the license agreement for using Pyannote. You need to have an account on Hugging Face and accept the license to use the models: https://huggingface.co/pyannote/speaker-diarization and https://huggingface.co/pyannote/segmentation. Get your KEY TOKEN here: https://hf.co/settings/tokens",
    "ht_token_ph" : "Token goes here...",
    "tab_docs" : "Document translation",
    "docs_input_label" : "Choose Document Source",
    "docs_input_info" : "It can be PDF, DOCX, TXT, or text",
    "docs_source_info" : "This is the original language of the text",
    "chunk_size_label" : "Max number of characters that the TTS will process per segment",
    "chunk_size_info" : "A value of 0 assigns a dynamic and more compatible value for the TTS.",
    "docs_button" : "Start Language Conversion Bridge",
    "cv_url_info" : "Automatically download the R.V.C. models from the URL. You can use links from HuggingFace or Drive, and you can include several links, each one separated by a comma. Example: https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.pth, https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.index",
    "replace_title" : "Replace voice: TTS to R.V.C.",
    "sec1_title" : "### 1. To enable its use, mark it as enable.",
    "enable_replace" : "Check this to enable the use of the models.",
    "sec2_title" : "### 2. Select a voice that will be applied to each TTS of each corresponding speaker and apply the configurations.",
    "sec2_subtitle" : "Depending on how many <TTS Speaker> you will use, each one needs its respective model. Additionally, there is an auxiliary one if for some reason the speaker is not detected correctly.",
    "cv_tts1" : "Choose the voice to apply for Speaker 1.",
    "cv_tts2" : "Choose the voice to apply for Speaker 2.",
    "cv_tts3" : "Choose the voice to apply for Speaker 3.",
    "cv_tts4" : "Choose the voice to apply for Speaker 4.",
    "cv_tts5" : "Choose the voice to apply for Speaker 5.",
    "cv_tts6" : "Choose the voice to apply for Speaker 6.",
    "cv_aux" : "- Voice to apply in case a Speaker is not detected successfully.",
    "cv_button_apply" : "APPLY CONFIGURATION",
    "tab_help" : "Help"
  },

  "spanish": {
    "description": """
        ### 🎥 **¡Traduce videos fácilmente con SoniTranslate!** 📽️

        Sube un video, audio o proporciona un enlace de YouTube. 📽️ **Obtén el cuaderno actualizado desde el repositorio oficial: [SoniTranslate](https://github.com/R3gm/SoniTranslate)!**

        Consulta la pestaña `Ayuda` para obtener instrucciones sobre cómo usarlo. ¡Comencemos a divertirnos con la traducción de videos! 🚀🎉
        """,
    "tutorial": """
        # 🔰 **Instrucciones de uso:**

        1. 📤 Sube un archivo de **video**, **audio** o proporciona un enlace de 🌐 **YouTube**.

        2. 🌍 Elige el idioma en el que deseas **traducir el video**.

        3. 🗣️ Especifica el **número de personas que hablan** en el video y **asigna a cada una una voz de texto a voz** adecuada para el idioma de traducción.

        4. 🚀 Presiona el botón '**Traducir**' para obtener los resultados.
        
        ---

        # 🧩 **SoniTranslate admite diferentes motores de TTS (Texto a Voz), los cuales son:**
        - EDGE-TTS → formato `en-AU-WilliamNeural-Male` → Rapidos y precisos.
        - FACEBOOK MMS → formato `en-facebook-mms VITS` → Voz más natural, por el momento solo usa CPU.
        - PIPER TTS → formato `en_US-lessac-high VITS-onnx` → Igual que el anterior, pero está optimizado tanto para CPU como para GPU.
        - BARK → formato `en_speaker_0-Male BARK` → De buena calidad pero lento y propenso a alucinaciones.
        - Coqui XTTS → formato `_XTTS_/AUTOMATIC.wav` → Solo disponible para Chinese (Simplified), English, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Spanish, Hungarian, Korean y Japanese.

        ---

        # 🎤 Cómo usar las voces R.V.C. y R.V.C.2 (Opcional) 🎶

        El objetivo es aplicar un R.V.C. al TTS (Texto a Voz) generado 🎙️

        1. En la pestaña `Voz Personalizada R.V.C.`, descarga los modelos que necesitas 📥 Puedes utilizar enlaces de Hugging Face y Google Drive en formatos como zip, pth o index. También puedes descargar repositorios completos de espacio HF, pero esta opción no es muy estable 😕

        2. Ahora, ve a `Reemplazar voz: TTS a R.V.C.` y marca la casilla `habilitar` ✅ Después de esto, puedes elegir los modelos que deseas aplicar a cada hablante de TTS 👩‍🦰👨‍🦱👩‍🦳👨‍🦲

        3. Ajusta el método F0 que se aplicará a todos los R.V.C. 🎛️

        4. Presiona `APLICAR CONFIGURACIÓN` para aplicar los cambios que hayas realizado 🔄

        5. Vuelve a la pestaña de traducción de video y haz clic en 'Traducir' ▶️ Ahora, la traducción se realizará aplicando el R.V.C. 🗣️

        Consejo: Puedes usar `Probar R.V.C.` para experimentar y encontrar el mejor TTS o configuraciones para aplicar al R.V.C. 🧪🔍
        
        ---
        
        """,
    "tab_translate" : "Traducción de video",
    "video_source": "Seleccionar Fuente de Video",
    "link_info": "Ejemplo: www.youtube.com/watch?v=g_9rPvbENUw",
    "link_ph": "Ingrese la URL aquí...",
    "dir_info": "Ejemplo: /usr/home/my_video.mp4",
    "dir_ph": "Ingrese la ruta aquí...",
    "sl_label": "Idioma de origen",
    "sl_info": "Este es el idioma original del video",
    "tat_label": "Traducir audio a",
    "tat_info": "Seleccione el idioma de destino y asegúrese también de seleccionar los TTS correspondientes a ese lenguaje.",
    "num_speakers": "Seleccione cuántas personas están hablando en el video.",
    "min_sk": "Mín. de hablantes",
    "max_sk": "Máx. de hablantes",
    "tts_select": "Seleccione la voz que desea para cada hablante.",
    "sk1": "TTS Hablante 1",
    "sk2": "TTS Hablante 2",
    "sk3": "TTS Hablante 3",
    "sk4": "TTS Hablante 4",
    "sk5": "TTS Hablante 5",
    "sk6": "TTS Hablante 6",
    "vc_title": "Imitación de voz en diferentes idiomas",
    "vc_subtitle": """
        ### Replicar la voz de una persona en varios idiomas.
        Si bien es efectiva con la mayoría de las voces cuando se usa adecuadamente, puede no alcanzar la perfección en todos los casos.
        La imitación de voz solo replica el tono del hablante de referencia, excluyendo el acento y la emoción, que son controlados por el modelo TTS del hablante base y no son replicados por el convertidor.
        Esto tomará muestras de audio del audio principal para cada hablante y las procesará.
        """,
    "vc_active_label": "Imitación de voz activa",
    "vc_active_info": "Imitación de voz activa: Replica el tono del hablante original",
    "vc_segments_label": "Máximo de muestras",
    "vc_segments_info": "Máximo de muestras: Es el número de muestras de audio que se generarán para el proceso, más es mejor pero puede agregar ruido",
    "vc_dereverb_label": "Dereverberación",
    "vc_dereverb_info": "Dereverberación: Aplica la dereverberación vocal a las muestras de audio.",
    "vc_remove_label": "Eliminar muestras anteriores",
    "vc_remove_info": "Eliminar muestras anteriores: Elimina las muestras generadas anteriormente, por lo que es necesario crear nuevas.",
    "xtts_title": "Crear un TTS basado en un audio",
    "xtts_subtitle": "Sube un archivo de audio de máximo 10 segundos con una voz. Utilizando XTTS, se creará un nuevo TTS con una voz similar al archivo de audio proporcionado.",
    "xtts_file_label": "Subir un breve audio con la voz",
    "xtts_name_label": "Nombre para el TTS",
    "xtts_name_info": "Usa un nombre sencillo",
    "xtts_dereverb_label": "Dereverberación del audio",
    "xtts_dereverb_info": "Dereverberación del audio: Aplica la dereverberación vocal al audio",
    "xtts_button": "Procesar el audio e incluirlo en el selector de TTS",
    "xtts_footer": "Generar voz XTTS automáticamente: Puedes usar '_XTTS_/AUTOMATIC.wav' en el selector de TTS para generar automáticamente segmentos para cada hablante al generar la traducción.",
    "extra_setting": "Configuraciones Avanzadas",
    "acc_max_label": "Máx. de Aceleración de Audio",
    "acc_max_info": "Aceleración máxima para segmentos de audio traducidos para evitar superposiciones. Un valor de 1.0 representa ninguna aceleración.",
    "aud_mix_label": "Método de Mezcla de Audio",
    "aud_mix_info": "Mezclar archivos de audio original y traducido para crear una salida personalizada y equilibrada con dos modos de mezcla disponibles.",
    "vol_ori": "Volumen audio original",
    "vol_tra": "Volumen audio traducido",
    "sub_type": "Tipo de Subtítulos",
    "whisper_title": "Configuración predeterminada de Whisper.",
    "srt_file_label" : "Subir un archivo de subtítulos SRT (Se utilizará en lugar de la transcripción de Whisper)",
    "voiceless_tk_label" : "Pista sin voz",
    "voiceless_tk_info" : "Pista sin voz: Elimina las voces originales del audio antes de combinarlo con el audio traducido. (Experimental)",
    "out_name_label": "Nombre del archivo",
    "out_name_info": "El nombre del archivo de salida",
    "preview_info": "La vista previa corta el video a solo 10 segundos con fines de prueba. Desactívelo para obtener la duración completa del video.",
    "edit_sub_label": "Editar subtítulos generados",
    "edit_sub_info": "Editar subtítulos generados: Permite ejecutar la traducción en 2 pasos. Primero, con el botón 'OBTENER SUBTÍTULOS Y EDITAR', obtiene los subtítulos para editarlos, y luego con el botón 'TRADUCIR', puede generar el video.",
    "button_subs": "OBTENER SUBTÍTULOS Y EDITAR",
    "editor_sub_label": "Subtítulos generados",
    "editor_sub_info": "Siéntase libre de editar el texto de los subtítulos generados aquí. Puede realizar cambios en las opciones de la interfaz antes de hacer clic en el botón 'TRADUCIR', excepto en 'Idioma de origen', 'Traducir audio a' y 'Máx. de hablantes', para evitar errores. Una vez que haya terminado, haga clic en el botón 'TRADUCIR'.",
    "editor_sub_ph": "Presione primero 'OBTENER SUBTÍTULOS Y EDITAR' para obtener los subtítulos",
    "button_translate": "TRADUCIR",
    "output_result_label": "DESCARGAR VIDEO TRADUCIDO",
    "sub_ori": "Subtítulos originales",
    "sub_tra": "Subtítulos traducidos",
    "ht_token_info": "Un paso importante es aceptar el acuerdo de licencia para usar Pyannote. Debe tener una cuenta en Hugging Face y aceptar la licencia para usar los modelos: https://huggingface.co/pyannote/speaker-diarization y https://huggingface.co/pyannote/segmentation. Obtenga su TOKEN aquí: https://hf.co/settings/tokens",
    "ht_token_ph": "Ingrese el token aquí...",
    "tab_docs": "Traducción de documento",
    "docs_input_label": "Elegir origen del documento",
    "docs_input_info": "Puede ser PDF, DOCX, TXT o texto",
    "docs_source_info": "Este es el idioma original del texto",
    "chunk_size_label" : "Máximo numero de caracteres que el TTS procesará por segmento.",
    "chunk_size_info" : "Un valor de 0 signa un valor dinámico y mejor combatible con el TTS.",
    "docs_button" : "Iniciar Puente de Conversión de Idioma",
    "cv_url_info": "Descargue automáticamente los modelos R.V.C. desde la URL. Puede utilizar enlaces de HuggingFace o Drive, e incluso puede incluir varios enlaces, cada uno separado por una coma. Ejemplo: https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.pth, https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.index",
    "replace_title": "Reemplazar voz: TTS a R.V.C.",
    "sec1_title": "### 1. Para habilitar su uso, márquelo como habilitado.",
    "enable_replace": "Marque esto para habilitar el uso de los modelos.",
    "sec2_title": "### 2. Seleccione una voz que se aplicará a cada TTS de cada hablante correspondiente y aplique las configuraciones.",
    "sec2_subtitle": "Dependiendo de cuántos <TTS Hablante> vaya a usar, cada uno necesita su respectivo modelo. Además, hay uno auxiliar si por alguna razón el hablante no es detectado correctamente.",
    "cv_tts1": "Voz a aplicar al TTS Hablante 1.",
    "cv_tts2": "Voz a aplicar al TTS Hablante 2.",
    "cv_tts3": "Voz a aplicar al TTS Hablante 3.",
    "cv_tts4": "Voz a aplicar al TTS Hablante 4.",
    "cv_tts5": "Voz a aplicar al TTS Hablante 5.",
    "cv_tts6": "Voz a aplicar al TTS Hablante 6.",
    "cv_aux": "- Voz a aplicar en caso de que un hablante no sea detectado correctamente.",
    "cv_button_apply": "APLICAR CONFIGURACIÓN",
    "tab_help": "Ayuda"
  }

}
