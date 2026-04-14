<?php

/**
 * Laravel Migration for ML API Integration
 * 
 * This migration creates the same tables in your Laravel database
 * so both Laravel and the ML API can share the same data.
 * 
 * Run: php artisan migrate
 */

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        // Crop Productions Table
        Schema::create('crop_productions', function (Blueprint $table) {
            $table->id();
            $table->string('municipality', 100)->index();
            $table->string('farm_type', 50)->index();
            $table->integer('year')->index();
            $table->string('month', 10)->index();
            $table->string('crop', 100)->index();
            $table->decimal('area_planted_ha', 10, 2);
            $table->decimal('area_harvested_ha', 10, 2)->nullable();
            $table->decimal('productivity_mt_ha', 10, 2)->nullable();
            $table->decimal('production_mt', 10, 2);
            $table->timestamps();
            
            // Composite indexes for common queries
            $table->index(['crop', 'municipality', 'year']);
            $table->index(['municipality', 'year', 'month']);
        });

        // Forecasts Table
        Schema::create('forecasts', function (Blueprint $table) {
            $table->id();
            $table->string('crop', 100)->index();
            $table->string('municipality', 100)->index();
            $table->integer('year')->index();
            $table->decimal('production_mt', 10, 2);
            $table->decimal('confidence_lower', 10, 2)->nullable();
            $table->decimal('confidence_upper', 10, 2)->nullable();
            $table->string('trend_direction', 20)->nullable();
            $table->decimal('growth_rate_percent', 5, 2)->nullable();
            $table->timestamp('generated_at')->useCurrent();
            $table->timestamps();
            
            $table->index(['crop', 'municipality', 'year']);
            $table->unique(['crop', 'municipality', 'year']);
        });

        // Prediction Logs Table
        Schema::create('prediction_logs', function (Blueprint $table) {
            $table->id();
            $table->string('municipality', 100);
            $table->string('farm_type', 50);
            $table->integer('year');
            $table->string('month', 10);
            $table->string('crop', 100);
            $table->decimal('area_planted_ha', 10, 2);
            $table->decimal('predicted_production_mt', 10, 2);
            $table->string('request_ip', 50)->nullable();
            $table->string('user_agent', 255)->nullable();
            $table->decimal('processing_time_ms', 10, 2)->nullable();
            $table->timestamp('created_at')->useCurrent();
            
            $table->index('created_at');
            $table->index('crop');
            $table->index('municipality');
        });

        // Model Metadata Table
        Schema::create('model_metadata', function (Blueprint $table) {
            $table->id();
            $table->string('version', 50)->unique();
            $table->string('model_type', 100);
            $table->timestamp('training_date');
            $table->decimal('test_score', 5, 4)->nullable();
            $table->decimal('cv_score', 5, 4)->nullable();
            $table->text('features_used')->nullable();
            $table->text('hyperparameters')->nullable();
            $table->text('performance_metrics')->nullable();
            $table->boolean('is_active')->default(true);
            $table->timestamps();
            
            $table->index('is_active');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('prediction_logs');
        Schema::dropIfExists('forecasts');
        Schema::dropIfExists('crop_productions');
        Schema::dropIfExists('model_metadata');
    }
};
