# Documentation Swagger

La documentation complète des endpoints de l'API est accessible ici :

<div id="swagger-ui"></div>

<script>
  const ui = SwaggerUIBundle({
    url: "static/swagger.json",
    dom_id: "#swagger-ui",
    deepLinking: true,
    presets: [
      SwaggerUIBundle.presets.apis,
      SwaggerUIStandalonePreset
    ],
    layout: "BaseLayout",
  })
</script>
