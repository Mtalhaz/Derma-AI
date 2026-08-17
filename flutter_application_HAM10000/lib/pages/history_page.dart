import 'dart:io';
import 'package:flutter/material.dart';
import '../database_helper.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({super.key});

  @override
  State<HistoryPage> createState() => HistoryPageState();
}

class HistoryPageState extends State<HistoryPage> {
  final DatabaseHelper _dbHelper = DatabaseHelper();
  List<Map<String, dynamic>> _lesions = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _loadHistory();
  }

  void refreshHistory() {
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    final data = await _dbHelper.getLesions();
    if (mounted) {
      setState(() {
        _lesions = data;
        _isLoading = false;
      });
    }
  }

  Future<void> _deleteLesion(String id) async {
    await _dbHelper.deleteLesion(id);
    _loadHistory();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Kayıt silindi.')),
      );
    }
  }

  String _getUrgencyLabel(String riskLevel, double confidence) {
    if (riskLevel == 'Yüksek Risk' && confidence > 70) return '⚠️ Acil Değerlendirme';
    if (riskLevel == 'Yüksek Risk') return '🔴 Dermatolog Önerilir';
    if (riskLevel == 'Orta Risk') return '🟡 Takip Edilmeli';
    return '🟢 Düşük Öncelik';
  }

  Color _getUrgencyColor(String riskLevel, double confidence) {
    if (riskLevel == 'Yüksek Risk' && confidence > 70) return Colors.red.shade700;
    if (riskLevel == 'Yüksek Risk') return Colors.redAccent;
    if (riskLevel == 'Orta Risk') return Colors.orangeAccent;
    return Colors.green;
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator(color: Colors.teal));
    }

    if (_lesions.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.history_rounded, size: 80, color: const Color(0xFF00E5FF).withOpacity(0.5)),
            const SizedBox(height: 16),
            const Text(
              "Henüz kaydedilmiş bir vakanız yok.",
              style: TextStyle(fontSize: 16, color: Colors.white70),
            ),
          ],
        ),
      );
    }

    return ListView.builder(
      padding: const EdgeInsets.all(12),
      itemCount: _lesions.length,
      itemBuilder: (context, index) {
        final lesion = _lesions[index];
        final date = DateTime.fromMillisecondsSinceEpoch(lesion['timestamp']);
        final day = "${date.day.toString().padLeft(2, '0')}/${date.month.toString().padLeft(2, '0')}/${date.year}";
        final time = "${date.hour.toString().padLeft(2, '0')}:${date.minute.toString().padLeft(2, '0')}";
        final confidence = lesion['confidence'] is double ? lesion['confidence'] : (lesion['confidence'] as num).toDouble();
        final riskLevel = lesion['risk_level'] ?? 'Düşük Risk';
        final urgencyLabel = _getUrgencyLabel(riskLevel, confidence);
        final urgencyColor = _getUrgencyColor(riskLevel, confidence);
        
        Color riskColor = riskLevel == 'Yüksek Risk' ? Colors.redAccent 
                          : riskLevel == 'Orta Risk' ? Colors.orangeAccent 
                          : const Color(0xFF00E5FF);

        return Card(
          elevation: 4,
          shadowColor: const Color(0xFF00E5FF).withOpacity(0.05),
          color: const Color(0xFF131B2F),
          margin: const EdgeInsets.symmetric(vertical: 8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
            side: BorderSide(color: const Color(0xFF00E5FF).withOpacity(0.15), width: 1)
          ),
          child: ExpansionTile(
            iconColor: const Color(0xFF00E5FF),
            collapsedIconColor: Colors.white70,
            leading: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.file(
                File(lesion['image_path']),
                width: 60,
                height: 60,
                fit: BoxFit.cover,
                errorBuilder: (ctx, err, stack) => const Icon(Icons.broken_image, size: 60, color: Colors.white24),
              ),
            ),
            title: Text(
              lesion['body_part'] ?? 'Bilinmiyor',
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: Colors.white),
            ),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 6),
                Wrap(
                  spacing: 8,
                  runSpacing: 4,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                      decoration: BoxDecoration(
                        color: riskColor.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: riskColor.withOpacity(0.3)),
                      ),
                      child: Text(riskLevel, style: TextStyle(color: riskColor, fontWeight: FontWeight.bold, fontSize: 11)),
                    ),
                    Text("$day $time", style: const TextStyle(fontSize: 12, color: Colors.white54)),
                  ],
                ),
              ],
            ),
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Urgency Badge
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                      decoration: BoxDecoration(
                        color: const Color(0xFF0A0E17),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: urgencyColor.withOpacity(0.3)),
                      ),
                      child: Text(urgencyLabel, style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13, color: urgencyColor)),
                    ),
                    const SizedBox(height: 16),
                    // Confidence Bar
                    Row(
                      children: [
                        const Text("Güven Oranı: ", style: TextStyle(color: Colors.white70, fontSize: 13)),
                        Expanded(
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(6),
                            child: LinearProgressIndicator(
                              value: confidence / 100,
                              minHeight: 8,
                              backgroundColor: Colors.white10,
                              valueColor: AlwaysStoppedAnimation<Color>(riskColor),
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Text("%${confidence.toStringAsFixed(1)}", style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: Colors.white)),
                      ],
                    ),
                    const SizedBox(height: 16),
                    // Bottom Row
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          "Sınıf: ${lesion['highest_risk_label'].toString().toUpperCase()}",
                          style: const TextStyle(fontSize: 13, color: Colors.white54, fontWeight: FontWeight.w500),
                        ),
                        IconButton(
                          icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                          onPressed: () => _deleteLesion(lesion['id']),
                          tooltip: "Geçmişten Sil",
                        )
                      ],
                    ),
                  ],
                ),
              )
            ],
          ),
        );
      },
    );
  }
}
