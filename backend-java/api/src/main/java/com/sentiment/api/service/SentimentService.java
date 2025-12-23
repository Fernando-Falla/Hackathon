package com.sentiment.api.service;

import com.sentiment.api.dto.SentimentResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Service
public class SentimentService {

    private final WebClient webClient;

    public SentimentService(WebClient.Builder webClientBuilder) {
        this.webClient = webClientBuilder.baseUrl("http://localhost:8080").build();
    }

    public SentimentResponse analyze(String text) {
        // Request body igual al que espera tu FastAPI
        String requestBody = "{\"text\": \"" + text + "\"}";

        return webClient.post()
                .uri("/predict/sentiment")
                .header("Content-Type", "application/json")
                .bodyValue(requestBody)
                .retrieve()
                .onStatus(HttpStatusCode::is5xxServerError, 
                    response -> response.bodyToMono(String.class)
                        .map(body -> new RuntimeException("Servicio ML no disponible: " + body)))
                .bodyToMono(SentimentResponse.class)
                .block(); // Si prefieren no reactivo, o usar .block()
    }
}
